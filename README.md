# PIV Turbulence Pipeline

Python pipeline for processing and analyzing stereo/planar PIV data from a Random Jet Array (RJA) facility generating homogeneous isotropic turbulence (HIT). General-purpose, facility-agnostic post-processing library — campaign-specific analysis (e.g. sunbathing-algorithm parameter sweeps) lives in its own downstream repo, such as [rja-hit-characterization](https://github.com/SPC3MN/rja-hit-characterization).

## Structure

```
decomposition/   Reynolds decomposition of raw PIV snapshots into mean + fluctuating fields, calculation and bootstrapping of mean/RMS velocities, structure functions, autocorrelations, and reynolds stresses
analysis/        Calculation of dissipation rate, integral length, Taylor microscale, Lumley anisotropy invariants (stereo), homogeneity/isotropy
jet_control/     Sunbathing-algorithm pump control-signal generation
plotting/        Shared plotting helpers and publication figure-size conventions
single_jet/      Single-jet characterization (vorticity, enstrophy, half-width, phase-locked development)
```

| File | Purpose |
|---|---|
| `decomposition/Planar_Decomposition.py` | Reynolds decomposition for 2D planar PIV data |
| `decomposition/Stereo_Decomposition.py` | Reynolds decomposition for stereo PIV data |
| `decomposition/Test_Decomposition.py` | Working/scratch copy of the decomposition pipeline |
| `analysis/Planar_Analysis.py` | Post-processing for planar PIV results: dissipation rate, integral length, homogeneity/isotropy and CIs. **Known bugs — see Issues.** |
| `analysis/Stereo_Analysis.py` | Post-processing for stereo PIV results: dissipation rate, integral length, Lumley anisotropy invariants, homogeneity/isotropy and CIs. **Known bugs — see Issues.** |
| `analysis/Anisotropy_Invariants.py` | Standalone Reynolds-stress / anisotropy-invariant / contour-centering helpers on npz input |
| `analysis/Bayesian_Bootstrap.py` | Bayesian (Dirichlet) bootstrap CIs; reads the `snap_%03d.npz` convention directly |
| `analysis/Statistics_Plotting.py` | Statistics figure generation |
| `jet_control/Sunbathing.py` | Generates/visualizes the sunbathing-algorithm pump on/off control signal |
| `plotting/Clean_Plotting.py` | Publication figure-size conventions (single-column, stacked, double-column) |
| `plotting/Plotting.py` | General plotting helpers |
| `single_jet/Single_Jet.py` | Single-jet characterization: vorticity, enstrophy, half-width, velocity/enstrophy profiles, phase-locked development |
| `single_jet/Test.py` | Working/scratch copy of the single-jet pipeline |

## Setup

```bash
pip install -r requirements.txt
```

Raw PIV data paths are currently hardcoded (e.g. `/Volumes/PIV Data1/...`) at the top of each script under a `# Control` section — update these to match your local data location before running.

**`analysis/*.py` requires scipy** (`scipy.special.kv`, `scipy.optimize.curve_fit`, `scipy.ndimage.gaussian_filter`) and will not import at all on a system where scipy's compiled extensions don't match the interpreter architecture (e.g. x86_64 wheels under an arm64 Python). `decomposition/*.py` has no such dependency. See [rja-hit-characterization](https://github.com/SPC3MN/rja-hit-characterization)'s `burst_piv.py` for a scipy-free reimplementation of the `analysis/` functionality, built for exactly that situation.

## Input formats

Both `decomposition/Planar_Decomposition.py` and `decomposition/Stereo_Decomposition.py` accept three input formats for `input_dir`, controlled by the `input_format` variable in each script's `# Control` section:

- `'csv'` — per-snapshot DaVis CSV export (`x [mm]`, `y [mm]`, `Velocity u/v[/w] [m/s]` columns), the original format.
- `'npz'` — per-snapshot `snap_*.npz` files (`X`, `Y`, `U`, `V`[, `W`] arrays), as produced by this pipeline's own `Save_NPZ` step, or by an external GPU-PIV pipeline that writes matching keys. Loading npz skips the CSV parse/pivot step entirely, so re-running a case from a previously exported `npz_dir` is significantly faster than re-parsing the original CSVs.
- `'vc7'` — reads DaVis vector data directly via [`lvpyio`](https://www.lavision.de/en/downloads/software/python_add_ons.php), with no CSV export step at all. `input_dir` may point to a `.set` file, or to a directory containing either a `.set` file or a flat folder of `.vc7` snapshots. `lvpyio` ships with the DaVis Python Add-Ons (LaVision download page) rather than PyPI, so it must be installed into the environment separately — the loader raises a clear `ImportError` with instructions if it isn't found. Since `lvpyio`'s exact attribute names can shift slightly between versions, if `_frame_components()` in `decomposition/Planar_Decomposition.py` / `decomposition/Stereo_Decomposition.py` raises against your installed version, inspect `frame.components` / `frame.scales` (or `dir(frame)`) and adjust the lookups there.
- `'auto'` (default) — detects the format from `input_dir`: `.set`/`.vc7` files (and no `.csv`/`.npz`) select `'vc7'`, `.npz` files (and no `.csv`) select `'npz'`, otherwise `'csv'`. A path directly to a `.set` file always selects `'vc7'`.
