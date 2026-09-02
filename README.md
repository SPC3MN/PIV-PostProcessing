# PIV Turbulence Pipeline

Python pipeline for processing and analyzing stereo/planar PIV data from a Random Jet Array (RJA) facility generating homogeneous isotropic turbulence (HIT).

## Structure

```
common/          Statistics/IO code shared by both decomposition and analysis
decomposition/   Reynolds decomposition of raw PIV snapshots into mean + fluctuating fields, calculation and bootstrapping of mean/RMS velocities, structure functions, autocorrelations, and reynolds stresses
analysis/        Calculation of dissipation rate, integral length, Lumley anisotropy invariants (stereo), homogeneity/isotropy
```

| File | Purpose |
|---|---|
| `decomposition/Planar_Decomposition.py` | Reynolds decomposition for 2D planar PIV data |
| `decomposition/Stereo_Decomposition.py` | Reynolds decomposition for stereo PIV data |
| `analysis/Planar_Analysis.py` | Post-processing for planar PIV results: calculation of dissipation rate, integral length, homogeneity/isotropy and the respective CIs |
| `analysis/Stereo_Analysis.py` | Post-processing for stereo PIV results: calculation of dissipation rate, integral length, Lumley anisotropy invariants, homogeneity/isotropy and the respective CIs |
| `common/discovery.py` | `discover_case_dirs` / `discover_case_dirs_or_root` — finds per-case subfolders under a root directory (the latter also accepts a root pointed directly at a single case folder) |
| `common/io_npz.py` | Loads a case's `snap_*.npz` snapshots, with an optional centered crop |
| `common/decomposition_stats.py` | Reynolds decomposition, bootstrap CIs, structure functions, autocorrelations, energy spectra — shared by both decomposition scripts |
| `common/anisotropy.py` | Reynolds stress tensor and Lumley anisotropy invariants (stereo) |
| `common/analysis_stats.py` | Taylor microscale, homogeneous-region detection, and the integral-length model fit — shared by both analysis scripts |
| `common/results_io.py` | Writes the collected per-case results to a single Excel workbook |

`analysis/` and `decomposition/` also contain a few pre-restructure scripts (`analysis/Anisotropy_Invariants.py`, `analysis/Bayesian_Bootstrap.py`, `analysis/Statistics_Plotting.py`, `decomposition/Test_Decomposition.py`) that predate `common/` and are not part of the maintained pipeline above — they're kept for reference but read their own ad hoc CSV/glob inputs rather than the `snap_*.npz` / `common.io_npz` convention.

## Setup

```bash
pip install -r requirements.txt
```

## Data layout

Decomposition and analysis each point at a **root directory of per-case subfolders** rather than a single hardcoded dataset:

```
<raw_root>/                      decomposition INPUT
  CaseA/  snap_000.npz  snap_001.npz  ...
  CaseB/  snap_000.npz  ...

<processed_root>/                decomposition OUTPUT == analysis INPUT
  CaseA/
    Ensemble_Averages/  Averages.npz  Bootstrapped_Statistics.npz
                         Structure_Function.npz  Autocorrelation_Function.npz
                         Energy_Spectra.npz  (planar only)
    Lumley_Statistics/  Lumley_Statistics.npz   (stereo only)
  CaseB/  ...

<results_path>                   analysis OUTPUT, e.g. results/Planar_Results.xlsx
```

Each `snap_*.npz` file holds `X`, `Y`, and the velocity components for one snapshot (`U`, `V` for planar; `U`, `V`, `W` for stereo).

Both stages auto-discover case folders via `common/discovery.py`, so adding a dataset is just adding a subfolder — no script edits needed to pick up a new case:

- **Decomposition** (`decomposition/Planar_Decomposition.py`, `decomposition/Stereo_Decomposition.py`) takes its input/output directories and every other option (snapshot limit, crop, autocorrelation/structure-function/spectra/bootstrap toggles, `only`) as **interactive prompts at runtime** — just run the script and answer the prompts, no editing required. `raw_root` may point either at one case folder of `snap_*.npz` files or at a parent folder of many case subfolders.
- **Analysis** (`analysis/Planar_Analysis.py`, `analysis/Stereo_Analysis.py`) still takes `processed_root`, `results_path`, `only`, and `save` as hardcoded variables in the `# Control` section at the top of the file — edit those before running. Set `only = "CaseName"` to limit a run to a single case, or leave it `None` to process every case folder found.

## Input format

Both `decomposition/Planar_Decomposition.py` and `decomposition/Stereo_Decomposition.py` load snapshots from `snap_*.npz` files (`X`, `Y`, `U`, `V`[, `W`] arrays) via `common/io_npz.py` — as produced by an upstream DaVis/PIV export step or a GPU-PIV pipeline that writes matching keys. There is no CSV or direct DaVis (`.set`/`.vc7`) loader in the maintained pipeline.

## Results

Each analysis script collects one summary row per case — means, RMS, isotropy, dissipation rate, integral length scales, Taylor microscale, and (stereo) Lumley anisotropy invariants — and, if `save = True` in its `# Control` section, writes the whole batch to a single Excel workbook (`results_path`) via `common/results_io.write_results_xlsx`. The workbook is regenerated from scratch each run, so it always reflects exactly the cases just processed rather than being incrementally appended to.
