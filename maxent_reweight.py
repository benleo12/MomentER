#!/usr/bin/env python3
"""
MaxEnt Reweight — Penalized Dual with Newton's Method
======================================================

Reweights a Monte Carlo prior to match analytic moment constraints using
maximum-entropy optimization. The dual problem is strictly convex, so
convergence is guaranteed.

Minimizes the penalized dual:

    L(λ) = log Z(λ) − Σ_k λ_k μ_k + ½ Σ_k σ_k² λ_k²

where Z(λ) = Σ_i w_i⁰ exp(Σ_k λ_k g_k(x_i))

Three modes:
  select — greedy forward selection of optimal moments
  run    — fit selected moments on full dataset, optionally with scale variations
  plot   — replot from saved lambdas (no Newton solve)

See README.md for usage.
"""

import os, glob, re, argparse, json, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ========================================
# Configuration
# ========================================
def get_args():
    p = argparse.ArgumentParser(
        description="MaxEnt reweighting of MC priors using analytic moments.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--mode", choices=["select", "run", "plot"], default="run",
                   help="select: find best moments.  run: fit on full dataset.  "
                        "plot: replot from saved lambdas.")

    # Required paths
    p.add_argument("--prior_dir", required=True,
                   help="Directory with prior MC files: dphi_values.csv.gz, "
                        "pT_values.csv.gz, m_values.csv.gz, pT_weight.csv.gz")
    p.add_argument("--mom_dir", required=True,
                   help="Directory with moment CSVs: DYMoments_<acc>.csv and "
                        "histogram CSVs for target distributions")
    p.add_argument("--accs", nargs="+", required=True,
                   help="Accuracy level(s), e.g. 'N2LLp+N2LO' or 'N4LLp+N3LO'")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for results")

    # Run/plot mode
    p.add_argument("--moments_file", default=None,
                   help="JSON with selected moments (from select mode output)")
    p.add_argument("--lambdas_json", default=None,
                   help="Lambdas JSON for plot mode (auto-detected if omitted)")

    # Optimization
    p.add_argument("--max_newton_steps", type=int, default=50)
    p.add_argument("--newton_tol", type=float, default=1e-9,
                   help="Convergence tolerance on max |gradient|")
    p.add_argument("--winsorize_pct", type=float, default=99.99,
                   help="Percentile cap for extreme feature values (0=off)")
    p.add_argument("--max_events", type=int, default=None,
                   help="Cap total events (for testing)")

    # Variation reweighting
    p.add_argument("--reweight_variations", action="store_true",
                   help="Reweight all scale variations (run mode)")
    p.add_argument("--n_workers", type=int, default=1,
                   help="Parallel workers (for select or variation reweighting)")

    # Select mode
    p.add_argument("--select_max_moments", type=int, default=30)
    p.add_argument("--select_n_events", type=int, default=1_000_000,
                   help="Events for selection trials (default: 1M)")
    p.add_argument("--select_max_k", type=int, default=5,
                   help="Max power of rT and dphi features")
    p.add_argument("--max_pair_power", type=int, default=6,
                   help="Maximum total power of a moment pair")

    # Plotting
    p.add_argument("--rebin_factor", type=int, default=3,
                   help="Rebin factor for plot histograms (default 3: 80→26 bins)")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args


# ========================================
# Divergence Metrics
# ========================================
def triangle_divergence(p, q, eps=1e-12):
    """TD(p||q) = Σ (p_i - q_i)² / (p_i + q_i)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    mask = np.isfinite(p) & np.isfinite(q) & ((p + q) > 0)
    p, q = p[mask], q[mask]
    return np.sum((p - q)**2 / (p + q + eps))


def chi2_divergence(p, q, eps=1e-12):
    p, q = np.asarray(p, float), np.asarray(q, float)
    mask = np.isfinite(p) & np.isfinite(q) & (q > eps)
    return np.sum((p[mask] - q[mask])**2 / q[mask])


def chi2_per_bin(p, q, q_unc, eps=1e-30):
    """Per-bin χ²: (1/N) Σ (p-q)²/σ². Returns (chi2/bin, n_bins)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    q_unc = np.asarray(q_unc, float)
    mask = np.isfinite(p) & np.isfinite(q) & np.isfinite(q_unc) & (q_unc > eps) & (q > 0)
    if mask.sum() == 0:
        return 0.0, 0
    p, q, q_unc = p[mask], q[mask], q_unc[mask]
    return np.sum((p - q)**2 / q_unc**2) / len(p), len(p)


# ========================================
# Load Prior Data
# ========================================
def load_prior(prior_dir):
    """Load prior MC events: dphi, pT, mass, weights."""
    print(f"\n[Loading Prior from {prior_dir}]")

    dphi = pd.read_csv(f"{prior_dir}/dphi_values.csv.gz").values.flatten()
    pT = pd.read_csv(f"{prior_dir}/pT_values.csv.gz").values.flatten()
    m = pd.read_csv(f"{prior_dir}/m_values.csv.gz").values.flatten()

    n = min(len(dphi), len(pT), len(m))
    dphi, pT, m = dphi[:n], pT[:n], m[:n]

    try:
        w = pd.read_csv(f"{prior_dir}/pT_weight.csv.gz").values.flatten()[:n].astype(np.float64)
    except Exception:
        w = np.ones(n, dtype=np.float64)

    good = np.isfinite(m) & (m > 1e-300)
    dphi, pT, m, w = dphi[good], pT[good], m[good], w[good]

    print(f"  Events: {n:,} total, {len(dphi):,} after filtering")

    d = (np.pi - dphi).astype(np.float64)
    rT = (pT / m).astype(np.float64)

    print(f"  d=(π−Δφ): mean={d.mean():.4f}, range=[{d.min():.4f}, {d.max():.4f}]")
    print(f"  rT=pT/m:  mean={rT.mean():.4f}, range=[{rT.min():.4f}, {rT.max():.4f}]")

    return {"d": d, "rT": rT, "pT": pT, "m": m, "w": w}


# ========================================
# Load Target Moments
# ========================================
def parse_moment(s):
    """Parse 'rt^1' or '(lnrt)^2' → (base, power)."""
    s = s.strip().lower()
    is_log = 'ln' in s
    if 'rt' in s:
        base = 'lnrt' if is_log else 'rt'
    elif 'dphi' in s:
        base = 'lndphi' if is_log else 'dphi'
    else:
        return None, 0
    m = re.search(r'\^(\d+)', s)
    k = int(m.group(1)) if m else 1
    return base, k


def load_moments(mom_path):
    """Load central moments with uncertainties from CSV."""
    print(f"\n[Loading Moments from {os.path.basename(mom_path)}]")
    df = pd.read_csv(mom_path)
    cols = {c.lower(): c for c in df.columns}
    c_fo = cols.get('scalefo') or cols.get('fo')
    c_res = cols.get('scaleres') or cols.get('res')
    c_o1, c_o2 = cols['o1'], cols['o2']

    value_col = None
    for name in ['value', 'val', 'moment']:
        if name in cols:
            value_col = cols[name]
            break
    if not value_col:
        value_col = df.columns[-2] if 'uncertainty' in cols else df.columns[-1]

    has_unc = 'uncertainty' in cols
    unc_col = cols.get('uncertainty')

    central = df[
        df[c_fo].str.contains('CV', case=False, na=False) &
        df[c_res].str.contains('CV', case=False, na=False)
    ].copy()
    if central.empty:
        raise RuntimeError(f"No central rows found in {mom_path}")

    moments, Z, Z_unc = [], None, None
    for _, row in central.iterrows():
        o1, o2 = str(row[c_o1]), str(row[c_o2])
        val = float(row[value_col])
        unc = float(row[unc_col]) if has_unc else None
        if 'rt^0' in o1.lower() and 'dphi^0' in o2.lower():
            Z, Z_unc = val, unc
        moments.append((o1, o2, val, unc))

    if Z is None:
        raise RuntimeError("Normalization moment (rt^0 × dphi^0) not found")

    moments_norm = []
    for o1, o2, val, unc in moments:
        val_norm = val / Z
        if unc is not None and Z_unc is not None and val != 0:
            unc_norm = abs(val_norm) * np.sqrt((unc/abs(val))**2 + (Z_unc/Z)**2)
        else:
            unc_norm = None
        moments_norm.append((o1, o2, val_norm, unc_norm))

    print(f"  Central moments: {len(moments_norm)}")
    return moments_norm


def get_scale_variations(mom_path):
    """Get all (FO, Res) scale variation pairs."""
    df = pd.read_csv(mom_path)
    cols = {c.lower(): c for c in df.columns}
    c_fo = cols.get('scalefo') or cols.get('fo')
    c_res = cols.get('scaleres') or cols.get('res')

    variations = df[[c_fo, c_res]].drop_duplicates().values.tolist()
    central, others = None, []
    for fo, res in variations:
        if 'CV' in str(fo).upper() and 'CV' in str(res).upper():
            central = (fo, res)
        else:
            others.append((fo, res))
    return central, others


def load_moments_for_scale(mom_path, scale_fo, scale_res):
    """Load moments for a specific scale variation."""
    df = pd.read_csv(mom_path)
    cols = {c.lower(): c for c in df.columns}
    c_fo = cols.get('scalefo') or cols.get('fo')
    c_res = cols.get('scaleres') or cols.get('res')
    c_o1, c_o2 = cols['o1'], cols['o2']

    value_col = None
    for name in ['value', 'val', 'moment']:
        if name in cols:
            value_col = cols[name]
            break
    if not value_col:
        value_col = df.columns[-2] if 'uncertainty' in cols else df.columns[-1]

    has_unc = 'uncertainty' in cols
    unc_col = cols.get('uncertainty')

    selected = df[(df[c_fo] == scale_fo) & (df[c_res] == scale_res)].copy()
    if selected.empty:
        return None

    moments, Z, Z_unc = [], None, None
    for _, row in selected.iterrows():
        o1, o2 = str(row[c_o1]), str(row[c_o2])
        val = float(row[value_col])
        unc = float(row[unc_col]) if has_unc else None
        if 'rt^0' in o1.lower() and 'dphi^0' in o2.lower():
            Z, Z_unc = val, unc
        moments.append((o1, o2, val, unc))

    if Z is None:
        return None

    moments_norm = []
    for o1, o2, val, unc in moments:
        val_norm = val / Z
        if unc is not None and Z_unc is not None and val != 0:
            unc_norm = abs(val_norm) * np.sqrt((unc/abs(val))**2 + (Z_unc/Z)**2)
        else:
            unc_norm = None
        moments_norm.append((o1, o2, val_norm, unc_norm))
    return moments_norm


# ========================================
# Build Features
# ========================================
def _is_rt_type(base):
    return base in ('rt', 'lnrt')

def _is_dphi_type(base):
    return base in ('dphi', 'lndphi')

def _compute_basis(values, base, k):
    if k == 0:
        return np.ones(len(values), dtype=np.float64)
    if base in ('rt', 'dphi'):
        return values ** k
    return np.log(np.maximum(values, 1e-30)) ** k

def _composite_key(b1, k1, b2, k2):
    parts = sorted([(b1, k1), (b2, k2)])
    return f"{parts[0][0]}^{parts[0][1]}*{parts[1][0]}^{parts[1][1]}"

def _display_name(f_name, g_name):
    fb, fk = f_name
    gb, gk = g_name
    if isinstance(fb, str) and '*' in fb and gb == 'const' and gk == 0:
        return fb.replace('*', '×')
    elif isinstance(gb, str) and '*' in gb and fb == 'const' and fk == 0:
        return gb.replace('*', '×')
    return f"{fb}^{fk}×{gb}^{gk}"

def _normalize_moment_name(name):
    parts = name.split('×')
    return '×'.join(sorted(parts)) if len(parts) == 2 else name


def build_features(prior, moments, max_k_rt, max_k_dphi, winsorize_pct=99.99):
    """Build F (rT) and G (dphi) feature matrices."""
    print(f"\n[Building Features: max_k_rt={max_k_rt}, max_k_dphi={max_k_dphi}]")

    rT = prior['rT'].astype(np.float64)
    d = prior['d'].astype(np.float64)

    if 0 < winsorize_pct < 100:
        rT_cap = np.percentile(rT, winsorize_pct)
        d_cap = np.percentile(d, winsorize_pct)
        rT = np.minimum(rT, rT_cap)
        d = np.minimum(d, d_cap)
        print(f"  Winsorized at {winsorize_pct}th percentile: rT<{rT_cap:.4f}, d<{d_cap:.4f}")

    need = {'rt': {0}, 'lnrt': {0}, 'dphi': {0}, 'lndphi': {0}}
    composite_rt, composite_dphi = set(), set()

    for o1, o2, _, _ in moments:
        b1, k1 = parse_moment(o1)
        b2, k2 = parse_moment(o2)
        if b1 is None or b2 is None:
            continue
        both_rt = _is_rt_type(b1) and _is_rt_type(b2)
        both_dphi = _is_dphi_type(b1) and _is_dphi_type(b2)
        if both_rt:
            if k1 > 0 and k2 > 0 and k1 <= max_k_rt and k2 <= max_k_rt:
                composite_rt.add((b1, k1, b2, k2))
            for base, k in [(b1, k1), (b2, k2)]:
                if k <= max_k_rt:
                    need[base].add(k)
        elif both_dphi:
            if k1 > 0 and k2 > 0 and k1 <= max_k_dphi and k2 <= max_k_dphi:
                composite_dphi.add((b1, k1, b2, k2))
            for base, k in [(b1, k1), (b2, k2)]:
                if k <= max_k_dphi:
                    need[base].add(k)
        else:
            for base, k in [(b1, k1), (b2, k2)]:
                max_k = max_k_rt if _is_rt_type(base) else max_k_dphi
                if k <= max_k:
                    need[base].add(k)

    # Build F (rT features)
    F_cols, F_names = [np.ones(len(rT))], [('const', 0)]
    for k in sorted(need['rt'] - {0}):
        F_cols.append(rT ** k); F_names.append(('rt', k))
    for k in sorted(need['lnrt'] - {0}):
        F_cols.append(np.log(np.maximum(rT, 1e-30)) ** k); F_names.append(('lnrt', k))
    for b1, k1, b2, k2 in sorted(composite_rt):
        F_cols.append(_compute_basis(rT, b1, k1) * _compute_basis(rT, b2, k2))
        F_names.append((_composite_key(b1, k1, b2, k2), 0))

    # Build G (dphi features)
    G_cols, G_names = [np.ones(len(d))], [('const', 0)]
    for k in sorted(need['dphi'] - {0}):
        G_cols.append(d ** k); G_names.append(('dphi', k))
    for k in sorted(need['lndphi'] - {0}):
        G_cols.append(np.log(np.maximum(d, 1e-30)) ** k); G_names.append(('lndphi', k))
    for b1, k1, b2, k2 in sorted(composite_dphi):
        G_cols.append(_compute_basis(d, b1, k1) * _compute_basis(d, b2, k2))
        G_names.append((_composite_key(b1, k1, b2, k2), 0))

    F = np.column_stack(F_cols)
    G = np.column_stack(G_cols)
    print(f"  F: {F.shape} features, G: {G.shape} features")
    return F, G, F_names, G_names


def extract_pairs(F_names, G_names, moments, max_k_rt, max_k_dphi):
    """Map moments to (i,j) feature index pairs."""
    print(f"\n[Extracting Moment Constraints]")

    F_idx = {name: i for i, name in enumerate(F_names)}
    G_idx = {name: j for j, name in enumerate(G_names)}

    moment_dict = {}
    for o1, o2, val, unc in moments:
        b1, k1 = parse_moment(o1)
        b2, k2 = parse_moment(o2)
        if b1 is None or b2 is None or (k1 == 0 and k2 == 0):
            continue

        both_rt = _is_rt_type(b1) and _is_rt_type(b2)
        both_dphi = _is_dphi_type(b1) and _is_dphi_type(b2)
        fi, gj = None, None

        if both_rt:
            key_f = (b1, k1) if k2 == 0 else ((b2, k2) if k1 == 0 else (_composite_key(b1, k1, b2, k2), 0))
            key_g = ('const', 0)
            fi, gj = F_idx.get(key_f), G_idx.get(key_g)
        elif both_dphi:
            key_f = ('const', 0)
            key_g = (b1, k1) if k2 == 0 else ((b2, k2) if k1 == 0 else (_composite_key(b1, k1, b2, k2), 0))
            fi, gj = F_idx.get(key_f), G_idx.get(key_g)
        elif _is_rt_type(b1):
            key_f = (b1, k1) if k1 > 0 else ('const', 0)
            key_g = (b2, k2) if k2 > 0 else ('const', 0)
            fi, gj = F_idx.get(key_f), G_idx.get(key_g)
        elif _is_rt_type(b2):
            key_f = (b2, k2) if k2 > 0 else ('const', 0)
            key_g = (b1, k1) if k1 > 0 else ('const', 0)
            fi, gj = F_idx.get(key_f), G_idx.get(key_g)

        if fi is not None and gj is not None:
            ij = (fi, gj)
            if ij not in moment_dict:
                moment_dict[ij] = {'vals': [], 'uncs': []}
            moment_dict[ij]['vals'].append(val)
            if unc is not None:
                moment_dict[ij]['uncs'].append(unc)

    pairs, targets, sigmas = [], [], []
    for (i, j), data in moment_dict.items():
        pairs.append((i, j))
        targets.append(np.mean(data['vals']))
        if data['uncs']:
            sigmas.append(np.sqrt(np.mean([u**2 for u in data['uncs']])) if len(data['uncs']) > 1 else data['uncs'][0])
        else:
            sigmas.append(None)

    print(f"  Constraints: {len(pairs)}")
    return (np.array(pairs, dtype=np.int64),
            np.array(targets, dtype=np.float64),
            sigmas)


# ========================================
# MaxEnt Model
# ========================================
class MaxEntDual:
    """Penalized dual MaxEnt with Newton's method."""

    def __init__(self, F, G, pairs, targets, w0, sigmas_target=None,
                 F_names=None, G_names=None):
        self.F, self.G = F, G
        self.pairs, self.targets = pairs, targets
        self.logw0 = np.log(np.maximum(w0, 1e-300))
        self.F_names, self.G_names = F_names, G_names
        self.K, self.N = len(pairs), F.shape[0]

        self._setup_sigmas(sigmas_target)
        self.lam = np.zeros(self.K, dtype=np.float64)
        _, self.m_prior, _ = self._compute_logZ_moments_cov(self.lam, need_cov=False)

        pulls = np.abs(self.m_prior - self.targets) / np.maximum(self.sigma, 1e-30)
        print(f"  Prior pulls: mean={pulls.mean():.2f}, max={pulls.max():.2f}")

    def _setup_sigmas(self, sigmas_target):
        if sigmas_target is not None:
            sigma_list = []
            for sig in sigmas_target:
                if sig is not None and sig > 0:
                    sigma_list.append(sig)
                else:
                    sigma_list.append(0.1 * abs(self.targets[len(sigma_list)]) + 1e-12)
            self.sigma = np.array(sigma_list, dtype=np.float64)
        else:
            self.sigma = 0.01 * np.abs(self.targets) + 1e-12
        self.sigma2 = self.sigma ** 2

    def _compute_logZ_moments_cov(self, lam, need_cov=True):
        batch = 500_000
        K, i_idx, j_idx = self.K, self.pairs[:, 0], self.pairs[:, 1]

        global_max = -np.inf
        for a in range(0, self.N, batch):
            b = min(a + batch, self.N)
            T = self.F[a:b, i_idx] * self.G[a:b, j_idx]
            m = (self.logw0[a:b] + T @ lam).max()
            if m > global_max:
                global_max = m

        sum_exp, S1 = 0.0, np.zeros(K, dtype=np.float64)
        S2 = np.zeros((K, K), dtype=np.float64) if need_cov else None

        for a in range(0, self.N, batch):
            b = min(a + batch, self.N)
            T = self.F[a:b, i_idx] * self.G[a:b, j_idx]
            w = np.exp(self.logw0[a:b] + T @ lam - global_max)
            sum_exp += w.sum()
            S1 += w @ T
            if need_cov:
                wT = w[:, None] * T
                S2 += wT.T @ T

        logZ = global_max + np.log(sum_exp)
        moments = S1 / sum_exp
        cov = (S2 / sum_exp - np.outer(moments, moments)) if need_cov else None
        return logZ, moments, cov

    def dual_loss(self, lam):
        logZ, _, _ = self._compute_logZ_moments_cov(lam, need_cov=False)
        return logZ - lam @ self.targets + 0.5 * (self.sigma2 * lam * lam).sum()

    def dual_loss_grad_hess(self, lam):
        logZ, moments, cov = self._compute_logZ_moments_cov(lam, need_cov=True)
        loss = logZ - lam @ self.targets + 0.5 * (self.sigma2 * lam * lam).sum()
        grad = moments - self.targets + self.sigma2 * lam
        hess = cov + np.diag(self.sigma2)
        return loss, grad, hess, moments

    def get_weights(self, lam=None):
        if lam is None:
            lam = self.lam
        batch = 500_000
        i_idx, j_idx = self.pairs[:, 0], self.pairs[:, 1]

        global_max = -np.inf
        for a in range(0, self.N, batch):
            b = min(a + batch, self.N)
            T = self.F[a:b, i_idx] * self.G[a:b, j_idx]
            m = (self.logw0[a:b] + T @ lam).max()
            if m > global_max:
                global_max = m

        sum_exp = 0.0
        for a in range(0, self.N, batch):
            b = min(a + batch, self.N)
            T = self.F[a:b, i_idx] * self.G[a:b, j_idx]
            sum_exp += np.exp(self.logw0[a:b] + T @ lam - global_max).sum()
        logZ = global_max + np.log(sum_exp)

        w = np.zeros(self.N, dtype=np.float64)
        for a in range(0, self.N, batch):
            b = min(a + batch, self.N)
            T = self.F[a:b, i_idx] * self.G[a:b, j_idx]
            w[a:b] = np.exp(self.logw0[a:b] + T @ lam - logZ)
        return w


# ========================================
# Newton Optimizer
# ========================================
def optimize_newton(model, max_steps=50, tol=1e-8, verbose=True):
    """Newton's method on the convex penalized dual."""
    lam = model.lam.copy()
    K = len(lam)
    print(f"\n[Newton Optimization: {K} constraints, tol={tol}]")

    for step in range(1, max_steps + 1):
        loss, grad, hess, moments = model.dual_loss_grad_hess(lam)
        grad_norm = np.max(np.abs(grad))
        resid = moments - model.targets
        pulls = resid / np.maximum(model.sigma, 1e-30)
        rms_pull = np.sqrt(np.mean(pulls**2))

        if verbose or step <= 3 or step % 5 == 0:
            print(f"  Step {step:3d}: Loss={loss:12.6f}  |∇|∞={grad_norm:.3e}  "
                  f"RMS pull={rms_pull:.4f}")

        if grad_norm < tol:
            print(f"  Converged at step {step}: |∇|∞ = {grad_norm:.3e}")
            break

        try:
            dlam = np.linalg.solve(hess + 1e-12 * np.eye(K), -grad)
        except np.linalg.LinAlgError:
            dlam = -grad / (np.diag(hess).max() + 1e-12)

        # Backtracking line search
        alpha, slope = 1.0, grad @ dlam
        if slope > 0:
            dlam = -grad / (np.diag(hess).max() + 1e-12)
            slope = grad @ dlam

        for _ in range(30):
            if model.dual_loss(lam + alpha * dlam) <= loss + 1e-4 * alpha * slope:
                break
            alpha *= 0.5

        lam = lam + alpha * dlam

    model.lam = lam

    loss, grad, _, moments = model.dual_loss_grad_hess(lam)
    pulls = (moments - model.targets) / np.maximum(model.sigma, 1e-30)
    print(f"\n  Final: Loss={loss:.6f}, RMS pull={np.sqrt(np.mean(pulls**2)):.4f}, "
          f"max|pull|={np.max(np.abs(pulls)):.4f}")
    return loss


# ========================================
# Triangle Divergence helpers
# ========================================
def compute_td(prior_data, w, target_dists, n_events):
    td = 0.0
    for dist_name, var_name in [('rTDist', 'rT'), ('dphiDist', 'd')]:
        if dist_name not in target_dists:
            continue
        edges = target_dists[dist_name]['edges']
        t_cen = target_dists[dist_name]['central']
        x = prior_data[var_name][:n_events]
        mask = (x >= edges[0]) & (x < edges[-1]) & np.isfinite(x)
        counts, _ = np.histogram(x[mask], bins=edges, weights=w[mask])
        widths = np.diff(edges)
        dens = counts / np.maximum(widths, 1e-300)
        dens = np.where(t_cen > 0, dens, 0.0)
        A_ref, A = np.sum(t_cen * widths), np.sum(dens * widths)
        if A > 0:
            dens *= A_ref / A
        td += triangle_divergence(dens, t_cen)
    return td


def compute_td_split(prior_data, w, target_dists, n_events):
    tds, chi2pbs = {}, {}
    for dist_name, var_name in [('rTDist', 'rT'), ('dphiDist', 'd')]:
        if dist_name not in target_dists:
            tds[dist_name] = chi2pbs[dist_name] = 0.0
            continue
        edges = target_dists[dist_name]['edges']
        t_cen = target_dists[dist_name]['central']
        t_unc = target_dists[dist_name].get('central_unc')
        x = prior_data[var_name][:n_events]
        mask = (x >= edges[0]) & (x < edges[-1]) & np.isfinite(x)
        counts, _ = np.histogram(x[mask], bins=edges, weights=w[mask])
        widths = np.diff(edges)
        dens = counts / np.maximum(widths, 1e-300)
        dens = np.where(t_cen > 0, dens, 0.0)
        A_ref, A = np.sum(t_cen * widths), np.sum(dens * widths)
        if A > 0:
            dens *= A_ref / A
        tds[dist_name] = triangle_divergence(dens, t_cen)
        chi2pbs[dist_name] = chi2_per_bin(dens, t_cen, t_unc)[0] if t_unc is not None else 0.0
    return (tds.get('rTDist', 0.0), tds.get('dphiDist', 0.0),
            chi2pbs.get('rTDist', 0.0), chi2pbs.get('dphiDist', 0.0))


def _feature_power(name_tuple):
    name, k = name_tuple
    if k > 0:
        return k
    if name == 'const':
        return 0
    powers = re.findall(r'\^(\d+)', name)
    return sum(int(p) for p in powers) if powers else 0

def _pair_total_power(pair, F_names, G_names):
    return _feature_power(F_names[pair[0]]) + _feature_power(G_names[pair[1]])


# ========================================
# Multiprocessing support
# ========================================
_MP_DATA = {}
_MP_VAR_DATA = {}


def _mp_var_init(F, G, w0, F_names, G_names, hist_edges_data,
                 prior_rT, prior_d, prior_pT, prior_m,
                 max_newton_steps, newton_tol):
    _MP_VAR_DATA.update({
        'F': F, 'G': G, 'w0': w0, 'F_names': F_names, 'G_names': G_names,
        'hist_edges_data': hist_edges_data,
        'prior_rT': prior_rT, 'prior_d': prior_d,
        'prior_pT': prior_pT, 'prior_m': prior_m,
        'max_newton_steps': max_newton_steps, 'newton_tol': newton_tol,
    })


def _mp_reweight_variation(task):
    import io, contextlib
    scale_fo, scale_res, pairs_arr, targets_arr, sigmas_list, warm_lam = task

    F, G, w0 = _MP_VAR_DATA['F'], _MP_VAR_DATA['G'], _MP_VAR_DATA['w0']
    F_names, G_names = _MP_VAR_DATA['F_names'], _MP_VAR_DATA['G_names']
    hist_edges_data = _MP_VAR_DATA['hist_edges_data']

    hist_edges = {}
    for key, edges in hist_edges_data.items():
        var_key = {'rT': 'prior_rT', 'dphi': 'prior_d', 'pT': 'prior_pT', 'mass': 'prior_m'}
        if key in var_key:
            hist_edges[key] = (_MP_VAR_DATA[var_key[key]], edges)

    with contextlib.redirect_stdout(io.StringIO()):
        model = MaxEntDual(F, G, pairs_arr, targets_arr, w0,
                           sigmas_target=sigmas_list,
                           F_names=F_names, G_names=G_names)
        if warm_lam is not None and len(warm_lam) == len(model.lam):
            model.lam[:] = warm_lam.copy()
        optimize_newton(model, max_steps=_MP_VAR_DATA['max_newton_steps'],
                        tol=_MP_VAR_DATA['newton_tol'], verbose=False)

    w_rew = model.get_weights()
    n_eff = 100 * (np.sum(w_rew)**2 / np.sum(w_rew**2)) / len(w_rew)
    _, m_var, _ = model._compute_logZ_moments_cov(model.lam, need_cov=False)
    pulls = (m_var - targets_arr) / np.maximum(model.sigma, 1e-30)
    rms = float(np.sqrt(np.mean(pulls**2)))

    hists = precompute_variation_hists(w_rew, hist_edges)
    return [((scale_fo, scale_res), hists, model.lam.copy(), rms, n_eff)]


def _mp_init_worker(F_sub, G_sub, w_sub, all_pairs, all_targets, all_sigmas,
                    F_names, G_names, prior_data, target_dists, n_ev):
    _MP_DATA.update({
        'F_sub': F_sub, 'G_sub': G_sub, 'w_sub': w_sub,
        'all_pairs': all_pairs, 'all_targets': all_targets, 'all_sigmas': all_sigmas,
        'F_names': F_names, 'G_names': G_names,
        'prior_data': prior_data, 'target_dists': target_dists, 'n_ev': n_ev,
    })


def _mp_eval_candidate(args):
    import io, contextlib
    idx, selected_, best_lam_, max_steps, tol = args

    d = _MP_DATA
    trial = selected_ + [idx]
    pairs_t = d['all_pairs'][trial]
    targets_t = d['all_targets'][trial]
    sigmas_t = [d['all_sigmas'][i_] for i_ in trial]

    with contextlib.redirect_stdout(io.StringIO()):
        model = MaxEntDual(d['F_sub'], d['G_sub'], pairs_t, targets_t, d['w_sub'],
                           sigmas_target=sigmas_t,
                           F_names=d['F_names'], G_names=d['G_names'])
        if len(selected_) > 0:
            model.lam[:len(selected_)] = best_lam_.copy()
        optimize_newton(model, max_steps=max_steps, tol=tol, verbose=False)

    w_rew = model.get_weights()
    w_rew = w_rew / w_rew.sum()
    td_rT, td_d, chi2pb_rT, chi2pb_d = compute_td_split(
        d['prior_data'], w_rew, d['target_dists'], d['n_ev'])
    neff = (np.sum(w_rew)**2 / np.sum(w_rew**2)) / d['n_ev']
    return (model.dual_loss(model.lam), td_rT + td_d, td_rT, td_d,
            chi2pb_rT, chi2pb_d, idx, model.lam.copy(), neff)


# ========================================
# Greedy Moment Selection
# ========================================
def greedy_select_by_td(F, G, all_pairs, all_targets, all_sigmas, w0,
                        prior_data, target_dists, F_names, G_names,
                        max_moments=30, n_events=1_000_000,
                        newton_steps=30, newton_tol=1e-7,
                        n_workers=1, min_improvement_pct=0.1,
                        min_steps=5, max_pair_power=6):
    """Greedy forward selection minimizing triangle divergence."""
    import io, contextlib

    n_ev = min(n_events, F.shape[0])
    F_sub, G_sub, w_sub = F[:n_ev], G[:n_ev], w0[:n_ev]

    n_candidates = len(all_pairs)
    available = list(range(n_candidates))
    selected = []

    w_prior_norm = w_sub / w_sub.sum()
    td_rT_0, td_d_0, chi2pb_rT_0, chi2pb_d_0 = compute_td_split(
        prior_data, w_prior_norm, target_dists, n_ev)
    td_baseline = td_rT_0 + td_d_0

    candidate_powers = {idx: _pair_total_power(all_pairs[idx], F_names, G_names)
                        for idx in available}

    print(f"\n{'='*80}")
    print(f"[Greedy Moment Selection]")
    print(f"  Candidates: {n_candidates}, Events: {n_ev:,}, Workers: {n_workers}")
    print(f"  Baseline TD: {td_baseline:.4f} (rT={td_rT_0:.4f}, dphi={td_d_0:.4f})")
    print(f"{'='*80}")

    best_lam = np.array([], dtype=np.float64)
    selection_log = []
    td_current = td_baseline
    screen_steps = 4

    _mp_init_worker(F_sub, G_sub, w_sub, all_pairs, all_targets,
                    all_sigmas, F_names, G_names, prior_data, target_dists, n_ev)

    _pool = None
    if n_workers > 1:
        import multiprocessing as mp
        import platform
        if platform.system() == 'Darwin':
            os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
        ctx = mp.get_context('fork')
        _pool = ctx.Pool(processes=n_workers, initializer=_mp_init_worker,
                         initargs=(F_sub, G_sub, w_sub, all_pairs, all_targets,
                                   all_sigmas, F_names, G_names, prior_data,
                                   target_dists, n_ev))

    def _run_batch(indices, selected_, best_lam_, max_steps, tol):
        tasks = [(idx, selected_, best_lam_, max_steps, tol) for idx in indices]
        if _pool is None or len(indices) < 4:
            return [_mp_eval_candidate(t) for t in tasks]
        results = []
        for r in _pool.imap_unordered(_mp_eval_candidate, tasks):
            results.append(r)
        return results

    for step in range(max_moments):
        if not available:
            break

        print(f"\n  --- Step {step+1}: {len(available)} candidates ---")

        # Screen then refine
        screen_results = _run_batch(available, selected, best_lam, screen_steps, newton_tol)
        screen_results.sort(key=lambda x: x[1])
        shortlist = [r[6] for r in screen_results[:min(50, len(screen_results))]]

        refine_results = _run_batch(shortlist, selected, best_lam, newton_steps, newton_tol)
        best_r = min(refine_results, key=lambda x: x[1])
        (best_dual, best_td, best_td_rT, best_td_d,
         best_chi2pb_rT, best_chi2pb_d,
         best_idx, best_lam_new, best_neff) = best_r

        i_, j_ = all_pairs[best_idx]
        name = _display_name(F_names[i_], G_names[j_])
        td_drop = td_current - best_td
        td_pct_cum = 100 * (1 - best_td / td_baseline)

        # Compute pulls
        trial = selected + [best_idx]
        pairs_t = all_pairs[trial]
        targets_t = all_targets[trial]
        sigmas_t = [all_sigmas[k] for k in trial]
        with contextlib.redirect_stdout(io.StringIO()):
            mc = MaxEntDual(F_sub, G_sub, pairs_t, targets_t, w_sub,
                            sigmas_target=sigmas_t, F_names=F_names, G_names=G_names)
        mc.lam = best_lam_new.copy()
        _, m_check, _ = mc._compute_logZ_moments_cov(mc.lam, need_cov=False)
        rms_pull = np.sqrt(np.mean(((m_check - targets_t) / np.maximum(mc.sigma, 1e-30))**2))

        pw = candidate_powers[best_idx]
        entry = {
            'step': step + 1, 'name': name, 'idx': int(best_idx), 'power': pw,
            'dual': float(best_dual), 'delta_dual': float(td_current - best_td),
            'td': float(best_td), 'td_rT': float(best_td_rT), 'td_dphi': float(best_td_d),
            'chi2pb_rT': float(best_chi2pb_rT), 'chi2pb_dphi': float(best_chi2pb_d),
            'delta_td': float(td_drop), 'pct_improvement': float(td_pct_cum),
            'rms_pull': float(rms_pull), 'neff': float(best_neff),
        }
        selection_log.append(entry)

        td_pct_step = 100 * td_drop / td_current if td_current > 0 else 0
        print(f"  {step+1:2d}. +{name:<25}  TD={best_td:.1f} "
              f"(rT={best_td_rT:.1f} d={best_td_d:.1f}) [{td_pct_cum:+.1f}%]  "
              f"N_eff={100*best_neff:.1f}%")

        # Stopping
        if step + 1 >= min_steps:
            if td_drop < 0:
                print(f"  TD got worse — stopping")
                selection_log.pop()
                break
            if td_pct_step < min_improvement_pct:
                print(f"  Saturated ({td_pct_step:.2f}% < {min_improvement_pct}%)")
                selected.append(best_idx)
                available.remove(best_idx)
                break

        selected.append(best_idx)
        available.remove(best_idx)
        td_current = best_td
        best_lam = best_lam_new

        # Backward step
        if len(selected) > 1:
            best_back_td, best_back_idx, best_back_lam = td_current, None, None
            for k, idx in enumerate(selected):
                trial_set = [x for x in selected if x != idx]
                with contextlib.redirect_stdout(io.StringIO()):
                    mb = MaxEntDual(F_sub, G_sub, all_pairs[trial_set],
                                    all_targets[trial_set], w_sub,
                                    sigmas_target=[all_sigmas[i_] for i_ in trial_set],
                                    F_names=F_names, G_names=G_names)
                    warm = np.delete(best_lam, k)
                    if len(warm) > 0:
                        mb.lam[:len(warm)] = warm
                    optimize_newton(mb, max_steps=newton_steps, tol=newton_tol, verbose=False)
                wb = mb.get_weights()
                wb = wb / wb.sum()
                td_b = compute_td(prior_data, wb, target_dists, n_ev)
                if td_b < best_back_td:
                    best_back_td, best_back_idx, best_back_lam = td_b, idx, mb.lam.copy()

            if best_back_idx is not None:
                rm_name = _display_name(F_names[all_pairs[best_back_idx][0]],
                                        G_names[all_pairs[best_back_idx][1]])
                selected.remove(best_back_idx)
                available.append(best_back_idx)
                td_current = best_back_td
                best_lam = best_back_lam
                print(f"      ← removed {rm_name:<25}  TD={td_current:.1f}")

    print(f"\n{'='*80}")
    print(f"[Selection Summary: {len(selected)} moments]")
    for e in selection_log:
        print(f"  {e['step']:2d}. {e['name']:<28} TD={e['td']:.1f} [{e['pct_improvement']:+.1f}%]")
    print(f"{'='*80}")

    if _pool is not None:
        _pool.close(); _pool.join()

    return selected, selection_log


# ========================================
# Load Target Distributions
# ========================================
def load_target_distributions(moments_csv, acc):
    """Load target histograms with statistical uncertainties."""
    if moments_csv is None:
        return {}

    from collections import defaultdict

    csv_files = list(moments_csv) if isinstance(moments_csv, (list, tuple)) else [moments_csv]

    rows = []
    for path in csv_files:
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        'dist': row['dist'], 'acc': row['acc'],
                        'fo': row['ScaleFO'], 'res': row['ScaleRes'],
                        'bin_lo': float(row['bin_lo']),
                        'bin_hi': float(row['bin_hi']),
                        'density': float(row.get('density', 'nan')),
                        'uncertainty': float(row.get('uncertainty', 'nan')) if 'uncertainty' in row else np.nan,
                    })
                except Exception:
                    continue

    if not rows:
        return {}

    acc_variants = {acc, acc.replace("'", "p"), acc.replace("'", ""), acc.replace("p", "'")}
    acc_rows = [r for r in rows if r['acc'] in acc_variants]
    if not acc_rows:
        return {}

    print(f"      Found {len(acc_rows)} rows for accuracy ~ '{acc}'")
    by_dist = defaultdict(lambda: defaultdict(list))
    for r in acc_rows:
        by_dist[r['dist']][(r['fo'], r['res'])].append(r)

    result = {}
    for dist_name in ['dphiDist', 'rTDist']:
        if dist_name not in by_dist:
            continue

        central_key = None
        for (fo, res) in by_dist[dist_name].keys():
            if str(fo).upper().startswith('CV') and str(res).upper().startswith('CV'):
                central_key = (fo, res)
                break
        if central_key is None:
            continue

        central_rows = sorted(by_dist[dist_name][central_key], key=lambda r: r['bin_lo'])
        edges = np.array([central_rows[0]['bin_lo']] + [r['bin_hi'] for r in central_rows], dtype=float)
        widths = np.diff(edges)

        central_dens = np.array([
            r['density'] if np.isfinite(r['density']) else r.get('count', 0) / max(bw, 1e-300)
            for r, bw in zip(central_rows, widths)
        ], dtype=float)
        central_unc = np.array([
            r['uncertainty'] if np.isfinite(r['uncertainty']) else 0.0
            for r in central_rows
        ], dtype=float)

        var_densities = []
        var_density_map = {central_key: central_dens}
        var_unc_map = {central_key: central_unc}

        for key, var_rows in by_dist[dist_name].items():
            if key == central_key:
                continue
            var_rows_sorted = sorted(var_rows, key=lambda r: r['bin_lo'])
            var_edges = np.array([var_rows_sorted[0]['bin_lo']] +
                                [r['bin_hi'] for r in var_rows_sorted], dtype=float)
            if len(var_edges) == len(edges) and np.allclose(var_edges, edges):
                var_dens = np.array([
                    r['density'] if np.isfinite(r['density']) else 0.0
                    for r in var_rows_sorted
                ], dtype=float)
                var_unc = np.array([
                    r['uncertainty'] if np.isfinite(r['uncertainty']) else 0.0
                    for r in var_rows_sorted
                ], dtype=float)
                var_densities.append(var_dens)
                var_density_map[key] = var_dens
                var_unc_map[key] = var_unc

        if var_densities:
            arr = np.vstack(var_densities)
            dens_min, dens_max = np.min(arr, axis=0), np.max(arr, axis=0)
        else:
            dens_min = dens_max = central_dens.copy()

        result[dist_name] = {
            'edges': edges, 'central': central_dens, 'central_unc': central_unc,
            'min': dens_min, 'max': dens_max,
            'var_map': var_density_map, 'var_unc_map': var_unc_map,
            'n_variations': len(var_densities),
        }
        rel_unc = 100 * central_unc / np.maximum(central_dens, 1e-300)
        print(f"      ✓ {dist_name}: {len(edges)-1} bins, {len(var_densities)} scale vars, "
              f"stat unc: {np.median(rel_unc):.1f}% median")

    return result


# ========================================
# Plotting
# ========================================
def hist_to_density(x, w, edges):
    x, w = np.asarray(x), np.asarray(w)
    mask = (x >= edges[0]) & (x < edges[-1]) & np.isfinite(x) & np.isfinite(w)
    counts, _ = np.histogram(x[mask], bins=edges, weights=w[mask])
    return counts / np.maximum(np.diff(edges), 1e-300)


def match_area(dens, ref_dens, edges, label=None):
    widths = np.diff(edges)
    A_ref, A = float(np.sum(ref_dens * widths)), float(np.sum(dens * widths))
    if A > 0:
        if label:
            print(f"    match_area [{label}]: MC={A:.4f}, Target={A_ref:.4f}, scale={A_ref/A:.4f}")
        dens = dens * (A_ref / A)
    return dens


def precompute_variation_hists(w_var, hist_edges):
    return {obs: hist_to_density(x, w_var, edges) for obs, (x, edges) in hist_edges.items()}


def rebin_density(dens, edges, factor=3):
    n = len(dens)
    n_new = n // factor
    widths = np.diff(edges)
    new_edges = np.empty(n_new + 1)
    new_dens = np.empty(n_new)
    for i in range(n_new):
        i0, i1 = i * factor, min((i + 1) * factor, n)
        new_edges[i] = edges[i0]
        new_dens[i] = np.sum(dens[i0:i1] * widths[i0:i1]) / np.sum(widths[i0:i1])
    new_edges[-1] = edges[min(n_new * factor, n)]
    return new_dens, new_edges


def plot_distributions(prior_data, w_prior, w_rew, acc, output_dir, moments_csv,
                       reweighted_dict=None, rebin_factor=3):
    """RIVET-style publication plots with grouped scale variation bands."""
    import matplotlib as mpl
    from matplotlib.ticker import AutoMinorLocator
    import shutil

    use_tex = shutil.which('latex') is not None
    mpl.rcParams.update({
        'text.usetex': use_tex, 'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'], 'font.size': 12,
        'axes.linewidth': 0.8, 'axes.labelsize': 14,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 5, 'ytick.major.size': 5,
        'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5,
        'legend.frameon': False, 'legend.fontsize': 9,
        'lines.linewidth': 1.2, 'savefig.bbox': 'tight', 'savefig.dpi': 300,
    })

    C = {'target': '#000000', 'target_stat': '#999999', 'target_scale': '#4682B4',
         'prior': '#D62728', 'rew': '#1F77B4'}

    VARIATION_GROUPS = {
        'resum': {'color': '#4682B4', 'label': r'Resum.\ scales'},
        'cs':    {'color': '#D95F02', 'label': r'CS kernel ($C_0^{\mathrm{NP}}$)'},
        'kappa': {'color': '#7570B3', 'label': r'$\kappa_{\mathrm{NP}}$'},
        'fo':    {'color': '#1B9E77', 'label': r'FO scales'},
    }

    def _classify(key):
        if isinstance(key, tuple):
            s = '/'.join(str(x) for x in key)
        else:
            s = str(key)
        su = s.upper()
        if 'CV->FO/CV->RES' in su or su == 'CENTRAL':
            return None
        if 'C0_NP' in su:
            return 'cs'
        if 'KAPPA_NP' in su:
            return 'kappa'
        if any(x in su for x in ['MUR->FO', 'MUF->FO', 'MURF->FO']):
            return 'fo'
        return 'resum'

    def _grouped_envelopes(obs_key, x_data, edges, ref_dens, var_dict,
                           area_target=None, var_map=None):
        groups = {}
        if not var_dict:
            return {}
        for sk, val in var_dict.items():
            grp = _classify(sk)
            if grp is None:
                continue
            d_v = val.get(obs_key) if isinstance(val, dict) else hist_to_density(x_data, val, edges)
            if d_v is None:
                continue
            d_v = np.where(ref_dens > 0, d_v, 0.0)
            if area_target is not None:
                t_var = var_map.get(sk[:2] if len(sk) > 2 else sk, area_target) if var_map else area_target
                d_v = match_area(d_v, t_var, edges)
            groups.setdefault(grp, []).append(d_v)
        return {g: (np.min(np.vstack(d), 0), np.max(np.vstack(d), 0), len(d))
                for g, d in groups.items()}

    def _target_envelopes(var_map):
        groups = {}
        if not var_map:
            return {}
        for sk, dens in var_map.items():
            grp = _classify(sk)
            if grp is None:
                continue
            groups.setdefault(grp, []).append(dens)
        return {g: (np.min(np.vstack(d), 0), np.max(np.vstack(d), 0), len(d))
                for g, d in groups.items()}

    def _step_xy(edges, vals):
        return np.repeat(edges, 2)[1:-1], np.repeat(vals, 2)

    def _step_fill(ax_, edges, lo, hi, **kw):
        x, y_lo = _step_xy(edges, lo)
        _, y_hi = _step_xy(edges, hi)
        ax_.fill_between(x, y_lo, y_hi, **kw)

    def _step_line(ax_, edges, vals, **kw):
        x, y = _step_xy(edges, vals)
        ax_.plot(x, y, **kw)

    def _save(fig_, name):
        for ext in ['pdf', 'png']:
            fig_.savefig(f"{output_dir}/{name}.{ext}",
                         dpi=300 if ext == 'pdf' else 200, bbox_inches='tight')
        plt.close(fig_)
        print(f"  Saved: {output_dir}/{name}.pdf (.png)")

    # Load targets
    target = load_target_distributions(moments_csv, acc) if moments_csv else {}
    acc_slug = acc.replace("'", "p").replace(" ", "")
    acc_label = acc.replace("'", r"$'$").replace("+", r"$+$")

    def _plot_observable(dist_key, var_name, ylabel, xlabel, plot_name):
        """Multi-panel plot for rT or dphi."""
        if dist_key not in target:
            return 0, 0
        edges = target[dist_key]['edges']
        t_cen = target[dist_key]['central']
        t_unc = target[dist_key]['central_unc']
        var_map = target[dist_key].get('var_map', {})

        d_prior = match_area(
            np.where(t_cen > 0, hist_to_density(prior_data[var_name], w_prior, edges), 0.0),
            t_cen, edges, label=f'{var_name} prior')
        d_rew = match_area(
            np.where(t_cen > 0, hist_to_density(prior_data[var_name], w_rew, edges), 0.0),
            t_cen, edges, label=f'{var_name} reweighted')

        td_p = triangle_divergence(d_prior, t_cen)
        td_r = triangle_divergence(d_rew, t_cen)
        chi2b_p, nb = chi2_per_bin(d_prior, t_cen, t_unc)
        chi2b_r, _ = chi2_per_bin(d_rew, t_cen, t_unc)
        print(f"  [TD] {var_name}: Prior={td_p:.6e}, Rew={td_r:.6e}")
        print(f"  [χ²/bin] {var_name}: Prior={chi2b_p:.2f}, Rew={chi2b_r:.2f} ({nb} bins)")

        rew_groups = _grouped_envelopes(var_name if var_name != 'd' else 'dphi',
                                        prior_data[var_name], edges, t_cen,
                                        reweighted_dict, area_target=t_cen, var_map=var_map)
        tgt_groups = _target_envelopes(var_map)

        # Rebin
        eo = edges.copy()
        t_cen_rb, edges_rb = rebin_density(t_cen, eo, factor=rebin_factor)
        t_unc_rb, _ = rebin_density(t_unc, eo, factor=rebin_factor)
        d_prior_rb, _ = rebin_density(d_prior, eo, factor=rebin_factor)
        d_rew_rb, _ = rebin_density(d_rew, eo, factor=rebin_factor)
        mask = t_cen_rb > 0

        for grp in list(rew_groups.keys()):
            lo, hi, n = rew_groups[grp]
            lo, _ = rebin_density(lo, eo, factor=rebin_factor)
            hi, _ = rebin_density(hi, eo, factor=rebin_factor)
            rew_groups[grp] = (lo, hi, n)
        for grp in list(tgt_groups.keys()):
            lo, hi, n = tgt_groups[grp]
            lo, _ = rebin_density(lo, eo, factor=rebin_factor)
            hi, _ = rebin_density(hi, eo, factor=rebin_factor)
            tgt_groups[grp] = (lo, hi, n)

        # Total envelopes
        tgt_lo = tgt_hi = t_cen_rb.copy()
        for lo, hi, _ in tgt_groups.values():
            tgt_lo = np.minimum(tgt_lo, lo)
            tgt_hi = np.maximum(tgt_hi, hi)
        rew_lo = rew_hi = d_rew_rb.copy()
        for lo, hi, _ in rew_groups.values():
            rew_lo = np.minimum(rew_lo, lo)
            rew_hi = np.maximum(rew_hi, hi)

        # Multi-panel figure
        sub_groups = ['resum', 'fo', 'cs', 'kappa']
        sub_labels = {'resum': r'Resum.', 'fo': r'FO',
                      'cs': r'$C_0^{\mathrm{NP}}$', 'kappa': r'$\kappa_{\mathrm{NP}}$'}
        n_sub = len(sub_groups)
        fig, axes = plt.subplots(
            2 + n_sub, 1, figsize=(6.5, 10),
            height_ratios=[3, 1.2] + [0.8]*n_sub, sharex=True,
            gridspec_kw={'hspace': 0.0})
        ax, axr = axes[0], axes[1]

        # Upper panel
        _step_line(ax, edges_rb, d_prior_rb, color=C['prior'], lw=1.2, alpha=0.7,
                   label=r'Prior (Sherpa)')
        _step_line(ax, edges_rb, d_rew_rb, color=C['rew'], lw=1.6, label=r'Reweighted')
        _step_line(ax, edges_rb, t_cen_rb, color=C['target'], lw=1.4,
                   label=r'Theory (central)')
        valid = t_cen_rb[mask]
        if len(valid):
            ax.set_ylim(valid.min() * 0.3, valid.max() * 5)
        ax.set_yscale('log')
        ax.set_ylabel(ylabel)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc='upper right', fontsize=9)
        ax.text(0.04, 0.96, r'$pp \to Z/\gamma^* \to \ell\ell$',
                transform=ax.transAxes, va='top', ha='left', fontsize=10)
        ax.text(0.04, 0.89, acc_label,
                transform=ax.transAxes, va='top', ha='left', fontsize=9, color='#555555')
        ax.text(0.04, 0.82,
                rf'$\chi^2/\mathrm{{bin}}$: {chi2b_p:.1f} $\to$ {chi2b_r:.1f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=8, color='#555555')

        # Main ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            r_prior = np.where(mask, d_prior_rb / t_cen_rb, 1.0)
            r_rew = np.where(mask, d_rew_rb / t_cen_rb, 1.0)
            r_unc = np.where(mask, t_unc_rb / t_cen_rb, 0.0)
            r_tlo = np.where(mask, tgt_lo / t_cen_rb, 1.0)
            r_thi = np.where(mask, tgt_hi / t_cen_rb, 1.0)
            r_rlo = np.where(mask, rew_lo / t_cen_rb, 1.0)
            r_rhi = np.where(mask, rew_hi / t_cen_rb, 1.0)

        _step_fill(axr, edges_rb, r_tlo, r_thi, color=C['target_scale'], alpha=0.20,
                   label=r'Theory unc.')
        _step_fill(axr, edges_rb, 1 - r_unc, 1 + r_unc, color=C['target_stat'], alpha=0.35)
        _step_fill(axr, edges_rb, r_rlo, r_rhi, color=C['rew'], alpha=0.15,
                   label=r'Rew.\ unc.')
        axr.axhline(1.0, color='black', ls='-', lw=0.6)
        _step_line(axr, edges_rb, r_prior, color=C['prior'], lw=1.2, alpha=0.7)
        _step_line(axr, edges_rb, r_rew, color=C['rew'], lw=1.6)
        axr.set_ylabel(r'Ratio', fontsize=9)
        axr.set_ylim(0.85, 1.15)
        axr.xaxis.set_minor_locator(AutoMinorLocator())
        axr.yaxis.set_minor_locator(AutoMinorLocator())
        axr.legend(loc='upper right', fontsize=7, ncol=2)

        # Sub-ratio panels
        for si, grp in enumerate(sub_groups):
            axs = axes[2 + si]
            gc = VARIATION_GROUPS[grp]
            if grp in tgt_groups:
                lo, hi, _ = tgt_groups[grp]
                with np.errstate(divide='ignore', invalid='ignore'):
                    _step_fill(axs, edges_rb,
                               np.where(mask, lo / t_cen_rb, 1.0),
                               np.where(mask, hi / t_cen_rb, 1.0),
                               color=gc['color'], alpha=0.20)
            if grp in rew_groups:
                lo, hi, _ = rew_groups[grp]
                with np.errstate(divide='ignore', invalid='ignore'):
                    _step_fill(axs, edges_rb,
                               np.where(mask, lo / t_cen_rb, 1.0),
                               np.where(mask, hi / t_cen_rb, 1.0),
                               color=gc['color'], alpha=0.35, hatch='///', linewidth=0)
            axs.axhline(1.0, color='black', ls='-', lw=0.5)
            _step_line(axs, edges_rb, r_rew, color=C['rew'], lw=1.0, alpha=0.5)
            axs.set_ylim(0.92, 1.08)
            axs.yaxis.set_minor_locator(AutoMinorLocator())
            axs.set_ylabel(sub_labels[grp], fontsize=8, rotation=0, labelpad=20, va='center')
            if si < n_sub - 1:
                axs.tick_params(labelbottom=False)

        axes[-1].set_xlabel(xlabel)
        axes[-1].xaxis.set_minor_locator(AutoMinorLocator())
        _save(fig, f'{plot_name}_{acc_slug}')
        return td_p, td_r

    # Plot rT and dphi with theory targets
    td_p_rT, td_r_rT = _plot_observable('rTDist', 'rT',
        r'$\mathrm{d}\sigma / \mathrm{d}r_T$', r'$r_T = p_T / m_{\ell\ell}$', 'rT')
    td_p_d, td_r_d = _plot_observable('dphiDist', 'd',
        r'$\mathrm{d}\sigma / \mathrm{d}(\pi - \Delta\phi)$', r'$\pi - \Delta\phi$', 'dphi')

    if td_p_rT + td_p_d > 0:
        print(f"  [TD] TOTAL: Prior={td_p_rT+td_p_d:.6e}, Rew={td_r_rT+td_r_d:.6e}")

    # pT plot (no theory target)
    if 'pT' in prior_data:
        pT_edges = np.array([0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,
                             60,80,100,150,200,300,500], dtype=float)
        pT_edges = pT_edges[pT_edges <= np.percentile(prior_data['pT'], 99.5)]
        if pT_edges[-1] < 200:
            pT_edges = np.append(pT_edges, 200)

        d_p = hist_to_density(prior_data['pT'], w_prior, pT_edges)
        d_r = match_area(hist_to_density(prior_data['pT'], w_rew, pT_edges),
                         d_p, pT_edges, label='pT reweighted')

        fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.5, 7.5), height_ratios=[3, 1],
                                      sharex=True, gridspec_kw={'hspace': 0.0})
        _step_line(ax, pT_edges, d_p, color=C['prior'], lw=1.4, alpha=0.85, label=r'Prior (Sherpa)')
        _step_line(ax, pT_edges, d_r, color=C['rew'], lw=1.6, label=r'Reweighted')
        ax.set_yscale('log'); ax.set_ylabel(r'$\mathrm{d}\sigma / \mathrm{d}p_T$ [GeV$^{-1}$]')
        ax.legend(loc='upper right'); ax.xaxis.set_minor_locator(AutoMinorLocator())
        mask_pT = d_p > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(mask_pT, d_r / d_p, 1.0)
        axr.axhline(1.0, color='black', ls='-', lw=0.6)
        _step_line(axr, pT_edges, r, color=C['rew'], lw=1.6)
        axr.set_ylabel(r'Rew.\ / Prior'); axr.set_xlabel(r'$p_T^{\ell\ell}$ [GeV]')
        axr.set_ylim(0.7, 1.3); axr.xaxis.set_minor_locator(AutoMinorLocator())
        axr.yaxis.set_minor_locator(AutoMinorLocator())
        _save(fig, f'pT_{acc_slug}')

    # Mass plot (no theory target)
    if 'm' in prior_data:
        mass_edges = np.linspace(50, 200, 31)
        d_p = hist_to_density(prior_data['m'], w_prior, mass_edges)
        d_r = match_area(hist_to_density(prior_data['m'], w_rew, mass_edges),
                         d_p, mass_edges, label='mass reweighted')

        fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.5, 7.5), height_ratios=[3, 1],
                                      sharex=True, gridspec_kw={'hspace': 0.0})
        _step_line(ax, mass_edges, d_p, color=C['prior'], lw=1.4, alpha=0.85, label=r'Prior (Sherpa)')
        _step_line(ax, mass_edges, d_r, color=C['rew'], lw=1.6, label=r'Reweighted')
        ax.set_yscale('log'); ax.set_ylabel(r'$\mathrm{d}\sigma / \mathrm{d}m_{\ell\ell}$ [GeV$^{-1}$]')
        ax.legend(loc='upper right'); ax.xaxis.set_minor_locator(AutoMinorLocator())
        mask_m = d_p > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(mask_m, d_r / d_p, 1.0)
        axr.axhline(1.0, color='black', ls='-', lw=0.6)
        _step_line(axr, mass_edges, r, color=C['rew'], lw=1.6)
        axr.set_ylabel(r'Rew.\ / Prior'); axr.set_xlabel(r'$m_{\ell\ell}$ [GeV]')
        axr.set_ylim(0.7, 1.3); axr.xaxis.set_minor_locator(AutoMinorLocator())
        axr.yaxis.set_minor_locator(AutoMinorLocator())
        _save(fig, f'mass_{acc_slug}')


def print_diagnostics(model, F_names, G_names, top_k=10):
    _, m_rew, _ = model._compute_logZ_moments_cov(model.lam, need_cov=False)
    targets, sigma = model.targets, model.sigma
    pull_rew = (m_rew - targets) / np.maximum(sigma, 1e-30)
    pct_err = 100 * (m_rew - targets) / np.maximum(np.abs(targets), 0.1 * sigma)
    idx_sort = np.argsort(np.abs(pull_rew))[::-1]

    print(f"\n{'Moment':<25} {'Target':>11} {'Reweighted':>11} {'%Err':>9} {'Pull':>7}")
    print("=" * 70)
    for i in idx_sort[:top_k]:
        i_f, j_g = model.pairs[i]
        name = _display_name(F_names[i_f], G_names[j_g])
        print(f"{name:<25} {targets[i]:11.5e} {m_rew[i]:11.5e} "
              f"{pct_err[i]:+8.2f}% {pull_rew[i]:+6.2f}")


# ========================================
# Helper utilities
# ========================================
def max_k_from_moment_names(names):
    max_rt = max_dphi = 0
    for name in names:
        for part in name.split('×'):
            m = re.match(r'(ln)?(rt|dphi)\^(\d+)', part.strip())
            if m:
                k = int(m.group(3))
                if m.group(2) == 'rt':
                    max_rt = max(max_rt, k)
                else:
                    max_dphi = max(max_dphi, k)
    return max_rt, max_dphi


def _find_hist_csvs(base_dir, acc_slug):
    all_csvs = glob.glob(os.path.join(base_dir, "*.csv"))
    main = [f for f in all_csvs if len(os.path.basename(f).split("__")) == 2]
    cand = [f for f in main if f"__{acc_slug}.csv" in os.path.basename(f)]
    return cand if cand else None


def _moment_matches_selection(o1, o2, name_set):
    b1, k1 = parse_moment(o1)
    b2, k2 = parse_moment(o2)
    if b1 is None or b2 is None:
        return False
    if k1 == 0 and k2 == 0:
        return True
    candidates = [f"{b1}^{k1}×{b2}^{k2}", f"{b2}^{k2}×{b1}^{k1}"]
    if k1 > 0 and k2 > 0:
        candidates.append(_composite_key(b1, k1, b2, k2).replace('*', '×'))
    if k1 == 0:
        candidates += [f"const^0×{b2}^{k2}", f"{b2}^{k2}×const^0"]
    if k2 == 0:
        candidates += [f"{b1}^{k1}×const^0", f"const^0×{b1}^{k1}"]
    return any(_normalize_moment_name(c) in name_set for c in candidates)


# ========================================
# Main
# ========================================
def main():
    args = get_args()

    print("=" * 80)
    print(f"MaxEnt Reweight — Mode: {args.mode.upper()}")
    print("=" * 80)

    prior = load_prior(args.prior_dir)

    if args.max_events and args.max_events < len(prior['rT']):
        N = args.max_events
        for key in prior:
            if hasattr(prior[key], '__len__') and len(prior[key]) > N:
                prior[key] = prior[key][:N].copy()
        print(f"  [Capped to {N:,} events]")

    for acc in args.accs:
        print(f"\n{'='*80}\nProcessing: {acc}\n{'='*80}")

        acc_slug = acc.replace("'", "p").replace(" ", "")
        mom_path = f"{args.mom_dir}/DYMoments_{acc_slug}.csv"
        if not os.path.exists(mom_path):
            print(f"  ERROR: {mom_path} not found"); continue

        moments = load_moments(mom_path)
        hist_csv = _find_hist_csvs(args.mom_dir, acc_slug)

        # ── SELECT MODE ──
        if args.mode == "select":
            max_k = args.select_max_k
            N_full = len(prior['w'])
            n_select = min(args.select_n_events, N_full)

            if n_select < N_full:
                rng = np.random.default_rng(42)
                idx_sub = np.sort(rng.choice(N_full, size=n_select, replace=False))
                prior_sel = {k: v[idx_sub] if hasattr(v, '__len__') and len(v) == N_full
                             else v for k, v in prior.items()}
            else:
                prior_sel = prior

            F, G, Fn, Gn = build_features(prior_sel, moments, max_k, max_k,
                                           winsorize_pct=args.winsorize_pct)
            pairs, targets, sigmas = extract_pairs(Fn, Gn, moments, max_k, max_k)

            target_dists = load_target_distributions(hist_csv, acc)
            if not target_dists:
                print("  ERROR: No target distributions found!"); continue

            # Filter trivial and high-power
            nontrivial = [i for i in range(len(pairs))
                          if _display_name(Fn[pairs[i][0]], Gn[pairs[i][1]]) != 'const^0×const^0'
                          and _pair_total_power(pairs[i], Fn, Gn) <= args.max_pair_power]
            all_pairs = pairs[nontrivial]
            all_targets = targets[nontrivial]
            all_sigmas = [sigmas[k] for k in nontrivial]

            print(f"  Candidates: {len(all_pairs)}")

            selected_idx, sel_log = greedy_select_by_td(
                F, G, all_pairs, all_targets, all_sigmas, prior_sel['w'],
                prior_sel, target_dists, Fn, Gn,
                max_moments=args.select_max_moments, n_events=n_select,
                newton_steps=args.max_newton_steps, newton_tol=args.newton_tol,
                n_workers=args.n_workers, max_pair_power=args.max_pair_power)

            selected_names = [_display_name(Fn[all_pairs[i][0]], Gn[all_pairs[i][1]])
                              for i in selected_idx]

            result = {
                'accuracy': acc,
                'selected_moments': selected_names,
                'selection_log': sel_log,
                'prior_td': {'total': float(sel_log[0]['td'] + sel_log[0]['delta_td']) if sel_log else 0},
                'final_td': float(sel_log[-1]['td']) if sel_log else 0,
                'n_events_used': n_select,
            }
            json_path = f"{args.output_dir}/selected_moments_{acc_slug}.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n  Saved: {json_path}")
            print(f"  Use with: --mode run --moments_file {json_path}")

        # ── RUN MODE ──
        elif args.mode == "run":
            if not args.moments_file:
                print("  ERROR: --moments_file required for run mode"); continue
            with open(args.moments_file) as f:
                sel_data = json.load(f)
            selected_names = set(sel_data['selected_moments'])
            print(f"  Loaded {len(selected_names)} moments from {args.moments_file}")

            use_k_rt, use_k_dphi = max_k_from_moment_names(selected_names)
            for name in selected_names:
                for part in name.split('×'):
                    m = re.match(r'ln(rt|dphi)\^(\d+)', part.strip())
                    if m:
                        k = int(m.group(2))
                        if m.group(1) == 'rt':
                            use_k_rt = max(use_k_rt, k)
                        else:
                            use_k_dphi = max(use_k_dphi, k)
            use_k_rt = max(use_k_rt, 1)
            use_k_dphi = max(use_k_dphi, 1)

            normalized = {_normalize_moment_name(n) for n in selected_names}
            moments_filt = [m for m in moments if _moment_matches_selection(m[0], m[1], normalized)]

            F, G, Fn, Gn = build_features(prior, moments_filt, use_k_rt, use_k_dphi,
                                           winsorize_pct=args.winsorize_pct)
            pairs, targets, sigmas = extract_pairs(Fn, Gn, moments_filt, use_k_rt, use_k_dphi)

            sel_p, sel_t, sel_s = [], [], []
            for idx, (i, j) in enumerate(pairs):
                if _normalize_moment_name(_display_name(Fn[i], Gn[j])) in normalized:
                    sel_p.append((i, j)); sel_t.append(targets[idx]); sel_s.append(sigmas[idx])

            if not sel_p:
                print("  ERROR: No constraints matched!"); continue

            pairs_arr = np.array(sel_p, dtype=np.int64)
            targets_arr = np.array(sel_t, dtype=np.float64)
            print(f"  Using {len(sel_p)} constraints")

            model = MaxEntDual(F, G, pairs_arr, targets_arr, prior['w'],
                               sigmas_target=sel_s, F_names=Fn, G_names=Gn)
            optimize_newton(model, max_steps=args.max_newton_steps,
                            tol=args.newton_tol, verbose=args.verbose)
            print_diagnostics(model, Fn, Gn, top_k=min(50, len(sel_p)))

            w_rew = model.get_weights()
            lam = model.lam.copy()
            n_eff = (np.sum(w_rew)**2 / np.sum(w_rew**2))
            print(f"\n  N_eff = {n_eff:.1f} ({100*n_eff/len(w_rew):.2f}%)")

            # Save weights
            pd.DataFrame({"w_rew": w_rew}).to_csv(
                f"{args.output_dir}/weights_{acc_slug}.csv.gz",
                index=False, compression="gzip")

            # Save lambdas
            lam_path = f"{args.output_dir}/lambdas_{acc_slug}.csv"
            with open(lam_path, 'w') as f:
                f.write("i,j,moment_name,lambda,sigma,target,pull\n")
                for p, (i, j) in enumerate(model.pairs):
                    name = _display_name(Fn[i], Gn[j])
                    pull = (model.m_prior[p] - model.targets[p]) / model.sigma[p]
                    f.write(f"{i},{j},{name},{model.lam[p]:.6e},{model.sigma[p]:.6e},"
                            f"{model.targets[p]:.6e},{pull:.3f}\n")

            # Variation reweighting
            reweighted_dict = None
            if args.reweight_variations:
                print(f"\n[Reweighting Scale Variations]")
                central_scale, var_scales = get_scale_variations(mom_path)
                target_dists = load_target_distributions(hist_csv, acc)

                hist_edges, hist_edges_data = {}, {}
                if 'rTDist' in target_dists:
                    hist_edges['rT'] = (prior['rT'], target_dists['rTDist']['edges'])
                    hist_edges_data['rT'] = target_dists['rTDist']['edges']
                if 'dphiDist' in target_dists:
                    hist_edges['dphi'] = (prior['d'], target_dists['dphiDist']['edges'])
                    hist_edges_data['dphi'] = target_dists['dphiDist']['edges']
                pT_edges = np.array([0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,
                                     60,80,100,150,200,300,500], dtype=float)
                pT_edges = pT_edges[pT_edges <= np.percentile(prior['pT'], 99.5)]
                if pT_edges[-1] < 200:
                    pT_edges = np.append(pT_edges, 200)
                hist_edges['pT'] = (prior['pT'], pT_edges)
                hist_edges_data['pT'] = pT_edges
                hist_edges['mass'] = (prior['m'], np.linspace(50, 200, 31))
                hist_edges_data['mass'] = np.linspace(50, 200, 31)

                reweighted_dict = {central_scale: precompute_variation_hists(w_rew, hist_edges)}
                all_lambdas = {central_scale: lam.copy()}

                var_tasks = []
                for scale_fo, scale_res in var_scales:
                    mvar = load_moments_for_scale(mom_path, scale_fo, scale_res)
                    if mvar is None:
                        continue
                    pv, tv, sv = extract_pairs(Fn, Gn, mvar, use_k_rt, use_k_dphi)
                    sp, st, ss = [], [], []
                    for idx, (i, j) in enumerate(pv):
                        if _normalize_moment_name(_display_name(Fn[i], Gn[j])) in normalized:
                            sp.append((i, j)); st.append(tv[idx]); ss.append(sv[idx])
                    if not sp:
                        continue
                    var_tasks.append((scale_fo, scale_res,
                                     np.array(sp, dtype=np.int64),
                                     np.array(st, dtype=np.float64),
                                     ss, lam.copy()))

                print(f"  {len(var_tasks)} variation tasks")

                _mp_var_init(F, G, prior['w'], Fn, Gn, hist_edges_data,
                             prior['rT'], prior['d'], prior.get('pT'), prior.get('m'),
                             args.max_newton_steps, args.newton_tol)

                for count, task in enumerate(var_tasks):
                    for key, hists, lam_v, rms_v, neff_v in _mp_reweight_variation(task):
                        reweighted_dict[key] = hists
                        all_lambdas[key] = lam_v
                        if isinstance(key, tuple) and len(key) == 2:
                            print(f"  {count+1}/{len(var_tasks)}: {key[0]}/{key[1]}  "
                                  f"RMS={rms_v:.4f}, N_eff={neff_v:.1f}%")

                print(f"\n  Reweighted {len(reweighted_dict)} variations")

                lam_path = f"{args.output_dir}/lambdas_all_variations_{acc_slug}.json"
                lam_save = {}
                for key, arr in all_lambdas.items():
                    k_str = '/'.join(str(k) for k in key) if isinstance(key, tuple) else str(key)
                    lam_save[k_str] = arr.tolist()
                with open(lam_path, 'w') as f:
                    json.dump(lam_save, f, indent=2)
                print(f"  Saved: {lam_path} ({len(all_lambdas)} sets)")

            plot_distributions(prior, prior['w'], w_rew, acc, args.output_dir,
                               hist_csv, reweighted_dict, rebin_factor=args.rebin_factor)

        # ── PLOT MODE ──
        elif args.mode == "plot":
            if not args.moments_file:
                print("  ERROR: --moments_file required"); continue
            with open(args.moments_file) as f:
                sel_data = json.load(f)
            selected_names = set(sel_data['selected_moments'])

            use_k_rt, use_k_dphi = max_k_from_moment_names(selected_names)
            for name in selected_names:
                for part in name.split('×'):
                    m = re.match(r'ln(rt|dphi)\^(\d+)', part.strip())
                    if m:
                        k = int(m.group(2))
                        if m.group(1) == 'rt':
                            use_k_rt = max(use_k_rt, k)
                        else:
                            use_k_dphi = max(use_k_dphi, k)
            use_k_rt, use_k_dphi = max(use_k_rt, 1), max(use_k_dphi, 1)

            normalized = {_normalize_moment_name(n) for n in selected_names}
            moments_filt = [m_ for m_ in moments if _moment_matches_selection(m_[0], m_[1], normalized)]

            F, G, Fn, Gn = build_features(prior, moments_filt, use_k_rt, use_k_dphi,
                                           winsorize_pct=args.winsorize_pct)
            pairs, targets, sigmas = extract_pairs(Fn, Gn, moments_filt, use_k_rt, use_k_dphi)

            sel_p, sel_t, sel_s = [], [], []
            for idx, (i, j) in enumerate(pairs):
                if _normalize_moment_name(_display_name(Fn[i], Gn[j])) in normalized:
                    sel_p.append((i, j)); sel_t.append(targets[idx]); sel_s.append(sigmas[idx])

            pairs_arr = np.array(sel_p, dtype=np.int64)
            targets_arr = np.array(sel_t, dtype=np.float64)

            lam_json = args.lambdas_json or f"{args.output_dir}/lambdas_all_variations_{acc_slug}.json"
            if not os.path.exists(lam_json):
                print(f"  ERROR: {lam_json} not found"); continue

            with open(lam_json) as f:
                all_lam = json.load(f)
            print(f"  Loaded {len(all_lam)} variation lambdas")

            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                model = MaxEntDual(F, G, pairs_arr, targets_arr, prior['w'],
                                   sigmas_target=sel_s, F_names=Fn, G_names=Gn)

            target_dists = load_target_distributions(hist_csv, acc)
            hist_edges = {}
            if 'rTDist' in target_dists:
                hist_edges['rT'] = (prior['rT'], target_dists['rTDist']['edges'])
            if 'dphiDist' in target_dists:
                hist_edges['dphi'] = (prior['d'], target_dists['dphiDist']['edges'])
            pT_edges = np.array([0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,
                                 60,80,100,150,200,300,500], dtype=float)
            pT_edges = pT_edges[pT_edges <= np.percentile(prior['pT'], 99.5)]
            if pT_edges[-1] < 200:
                pT_edges = np.append(pT_edges, 200)
            hist_edges['pT'] = (prior['pT'], pT_edges)
            hist_edges['mass'] = (prior['m'], np.linspace(50, 200, 31))

            reweighted_dict, w_rew = {}, None
            for key_str, lam_list in all_lam.items():
                lam_v = np.array(lam_list, dtype=np.float64)
                if len(lam_v) != len(model.lam):
                    continue
                w_v = model.get_weights(lam_v)
                reweighted_dict[tuple(key_str.split('/')) if '/' in key_str else key_str] = \
                    precompute_variation_hists(w_v, hist_edges)
                if 'CV->FO' in key_str and 'CV->Res' in key_str:
                    w_rew = w_v
                del w_v

            print(f"  Reconstructed {len(reweighted_dict)} variations")
            if w_rew is not None:
                n_eff = (np.sum(w_rew)**2 / np.sum(w_rew**2))
                print(f"  N_eff = {n_eff:.1f} ({100*n_eff/len(w_rew):.2f}%)")

            plot_distributions(prior, prior['w'], w_rew, acc, args.output_dir,
                               hist_csv, reweighted_dict, rebin_factor=args.rebin_factor)

    print(f"\n{'='*80}\nCOMPLETE\n{'='*80}")


if __name__ == "__main__":
    main()
