# MaxEnt Reweight

Reweights Monte Carlo event samples to match analytic moment constraints using
maximum-entropy (MaxEnt) optimization. The method solves a strictly convex
penalized dual problem with Newton's method, guaranteeing convergence.

## Method

Given a prior MC sample with weights $w_i^0$, we find new weights

$$w_i = \frac{w_i^0 \exp\bigl(\sum_k \lambda_k\, g_k(x_i)\bigr)}{Z(\lambda)}$$

that match a set of target moments $\mu_k$ computed from analytic resummation,
while staying as close as possible (in KL divergence) to the prior.

The Lagrange multipliers $\lambda_k$ are found by minimizing the penalized dual:

$$L(\lambda) = \log Z(\lambda) - \sum_k \lambda_k \mu_k + \tfrac{1}{2}\sum_k \sigma_k^2 \lambda_k^2$$

where $\sigma_k$ are the target moment uncertainties (regularization).

## Installation

```bash
pip install numpy pandas matplotlib
```

No other dependencies. Python 3.9+.

## What's Included

The `moments/` directory ships with analytic theory predictions at four
accuracy levels for Drell-Yan at 13.6 TeV:

| Accuracy | Moments | rT distribution | dphi distribution | Pre-selected |
|----------|---------|-----------------|-------------------|--------------|
| N1LL'+N1LO | `DYMoments_N1LLppN1LO.csv` | `rTDist__N1LLppN1LO.csv` | `dphiDist__N1LLppN1LO.csv` | — |
| N2LL'+N2LO | `DYMoments_N2LLp+N2LO.csv` | `rTDist__N2LLp+N2LO.csv` | `dphiDist__N2LLp+N2LO.csv` | `selected_moments_N2LLp+N2LO.json` |
| N3LL'+N3LO | `DYMoments_N3LLppN3LO.csv` | `rTDist__N3LLppN3LO.csv` | `dphiDist__N3LLppN3LO.csv` | — |
| N4LL'+N3LO | `DYMoments_N4LLp+N3LO.csv` | `rTDist__N4LLp+N3LO.csv` | `dphiDist__N4LLp+N3LO.csv` | `selected_moments_N4LLp+N3LO.json` |

Each moment file contains central values and 28 scale variations. The
pre-selected moment JSONs can be used directly with `--mode run` (step 2
below), skipping the selection step.

**You only need to provide your own MC prior** (see [Input Data Format](#input-data-format)).

## Quick Start

### 1. Select optimal moments (optional)

Pre-selected moments are provided for N2LL'+N2LO and N4LL'+N3LO. Run this
step only if you want to re-optimize the selection for a different prior or
accuracy level.

```bash
python maxent_reweight.py \
  --mode select \
  --prior_dir /path/to/your_prior \
  --mom_dir moments \
  --accs "N2LLp+N2LO" \
  --output_dir output/select_N2LLp \
  --select_n_events 1000000 \
  --n_workers 4
```

Output: `selected_moments_N2LLp+N2LO.json`

### 2. Run full reweighting

```bash
python maxent_reweight.py \
  --mode run \
  --prior_dir /path/to/your_prior \
  --mom_dir moments \
  --accs "N4LLp+N3LO" \
  --moments_file moments/selected_moments_N4LLp+N3LO.json \
  --output_dir output/run_N4LLp \
  --reweight_variations
```

This fits the selected moments on the full dataset and reweights all 28 scale
variations. Output:

- `weights_N4LLp+N3LO.csv.gz` — per-event reweighted weights
- `lambdas_N4LLp+N3LO.csv` — fitted Lagrange multipliers
- `lambdas_all_variations_N4LLp+N3LO.json` — lambdas for all scale variations
- `rT_N4LLp+N3LO.pdf` / `.png` — rT distribution with uncertainty bands
- `dphi_N4LLp+N3LO.pdf` / `.png` — dphi distribution with uncertainty bands
- `pT_N4LLp+N3LO.pdf` / `.png` — pT distribution (reweighted vs prior)
- `mass_N4LLp+N3LO.pdf` / `.png` — invariant mass distribution

### 3. Replot from saved lambdas

```bash
python maxent_reweight.py \
  --mode plot \
  --prior_dir /path/to/your_prior \
  --mom_dir moments \
  --accs "N4LLp+N3LO" \
  --moments_file moments/selected_moments_N4LLp+N3LO.json \
  --lambdas_json output/run_N4LLp/lambdas_all_variations_N4LLp+N3LO.json \
  --output_dir output/plots_N4LLp
```

Reconstructs weights from saved lambdas and regenerates plots without
re-running the Newton solver. Useful for tweaking plot style.

## Input Data Format

### Prior MC files (`--prior_dir`) — you provide this

Four gzipped CSV files, one column each, one row per event:

| File | Content |
|------|---------|
| `dphi_values.csv.gz` | Dilepton azimuthal opening angle $\Delta\phi$ |
| `pT_values.csv.gz` | Dilepton transverse momentum $p_T$ [GeV] |
| `m_values.csv.gz` | Dilepton invariant mass $m_{\ell\ell}$ [GeV] |
| `pT_weight.csv.gz` | Event weight (optional, defaults to 1) |

### Theory moments and distributions (`--mom_dir`) — included in `moments/`

These are provided in the repository. The format is documented below for
reference (e.g., if you want to add a new accuracy level).

**Moments CSV** (`DYMoments_<acc>.csv`):

| Column | Description |
|--------|-------------|
| `ScaleFO` | Fixed-order scale choice (`CV->FO` = central) |
| `ScaleRes` | Resummation scale choice (`CV->Res` = central) |
| `O1` | First observable (e.g., `rt^1`, `(lnrt)^2`) |
| `O2` | Second observable (e.g., `dphi^1`, `const`) |
| `Value` | Moment value $\langle O_1 \times O_2 \rangle$ |
| `Uncertainty` | Statistical uncertainty |

**Histogram CSV** (`rTDist__<acc>.csv`, `dphiDist__<acc>.csv`):

| Column | Description |
|--------|-------------|
| `dist` | Distribution name: `rTDist` or `dphiDist` |
| `acc` | Accuracy label |
| `ScaleFO`, `ScaleRes` | Scale choice |
| `bin_lo`, `bin_hi` | Bin edges |
| `density` | Differential cross section density |
| `uncertainty` | Statistical uncertainty on density |

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `run` | `select`, `run`, or `plot` |
| `--prior_dir` | *required* | Path to prior MC files |
| `--mom_dir` | *required* | Path to moment/histogram CSVs |
| `--accs` | *required* | Accuracy level(s) |
| `--output_dir` | *required* | Output directory |
| `--moments_file` | — | Selected moments JSON (run/plot mode) |
| `--lambdas_json` | — | Saved lambdas JSON (plot mode) |
| `--reweight_variations` | off | Reweight all scale variations |
| `--n_workers` | 1 | Parallel workers |
| `--select_max_moments` | 30 | Max moments to select |
| `--select_n_events` | 1M | Events for selection |
| `--select_max_k` | 5 | Max power of features |
| `--max_pair_power` | 6 | Max total power of moment pair |
| `--max_newton_steps` | 50 | Newton iterations |
| `--newton_tol` | 1e-9 | Convergence tolerance |
| `--winsorize_pct` | 99.99 | Feature winsorization percentile |
| `--max_events` | — | Cap events (for testing) |
| `--rebin_factor` | 3 | Plot rebinning (80→26 bins) |
| `--verbose` | off | Verbose Newton output |

## Output Plots

The rT and dphi plots use a multi-panel layout:

1. **Upper panel**: Prior, Reweighted, and Theory (central) distributions
2. **Main ratio**: Total theory and reweighted uncertainty envelopes
3. **Sub-ratio panels**: Decomposed by uncertainty source
   - Resummation scales
   - Fixed-order scales
   - CS kernel ($C_0^{\mathrm{NP}}$)
   - Non-perturbative ($\kappa_{\mathrm{NP}}$)

Solid bands = theory uncertainty, hatched bands = reweighted uncertainty.
