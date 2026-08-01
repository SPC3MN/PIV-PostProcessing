# PIV Turbulence Pipeline

Python pipeline for processing and analyzing stereo/planar PIV data from a Random Jet Array (RJA) facility generating homogeneous isotropic turbulence (HIT).

## Structure

```
decomposition/   Reynolds decomposition of raw PIV snapshots into mean + fluctuating fields, calculation and bootstrapping of mean/RMS velocities, structure functions, autocorrelations, and reynolds stresses
analysis/  Calculation of dissipation rate, integral length, Lumley anisotropy invariants (stereo), homogeneity/isotropy
```

| File | Purpose |
|---|---|
| `decomposition/Planar_Decomposition.py` | Reynolds decomposition for 2D planar PIV data |
| `decomposition/Stereo_Decomposition.py` | Reynolds decomposition for stereo PIV data |
| `analysis/Planar_Analysis.py` | Post-processing for planar PIV results: calculation of dissipation rate, integral length, homogeneity/isotropy and the respective CIs |
| `analysis/Stereo_Analysis.py` | Post-processing for stereo PIV results: calculation of dissipation rate, integral length, Lumley anisotropy invariants, homogeneity/isotropy and the respective CIs |

## Setup

```bash
pip install -r requirements.txt
```

Raw PIV data paths are currently hardcoded (e.g. `/Volumes/PIV Data1/...`) at the top of each script under a `# Control` section — update these to match your local data location before running.

## Input formats

Both `decomposition/Planar_Decomposition.py` and `decomposition/Stereo_Decomposition.py` accept two input formats for `input_dir`, controlled by the `input_format` variable in each script's `# Control` section:

- `'csv'` — per-snapshot DaVis CSV export (`x [mm]`, `y [mm]`, `Velocity u/v[/w] [m/s]` columns), the original format.
- `'npz'` — per-snapshot `snap_*.npz` files (`X`, `Y`, `U`, `V`[, `W`] arrays), as produced by this pipeline's own `Save_NPZ` step, or by an external GPU-PIV pipeline that writes matching keys. Loading npz skips the CSV parse/pivot step entirely, so re-running a case from a previously exported `npz_dir` is significantly faster than re-parsing the original CSVs.
- `'auto'` (default) — detects the format from the files present in `input_dir`: uses `'npz'` if the directory contains `.npz` files and no `.csv` files, otherwise `'csv'`.
