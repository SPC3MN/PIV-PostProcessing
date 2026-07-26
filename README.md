# PIV Turbulence Pipeline

Python pipeline for processing and analyzing stereo/planar PIV data from a Random Jet Array (RJA) facility generating homogeneous isotropic turbulence (HIT).

## Structure

```
decomposition/   Reynolds/triple decomposition of raw PIV snapshots into mean + fluctuating fields
analysis/        Structure functions, autocorrelations, dissipation rate, Lumley anisotropy invariants, Bayesian bootstrap CIs
plotting/        Figure generation from processed statistics (publication-style formatting)
jet_control/     Pump on/off control signal generation for pulsed jet forcing (sunbathing algorithm)
single_jet/      Single-jet velocity field statistics (spreading rate, KE, enstrophy)
```

| File | Purpose |
|---|---|
| `decomposition/Planar_Decomposition.py` | Reynolds decomposition for 2D planar PIV data |
| `decomposition/Stereo_Decomposition.py` | Reynolds decomposition for stereo PIV data |
| `analysis/Planar_Analysis.py` | Post-processing for planar PIV results: mean/RMS velocities, structure functions, dissipation rate, integral length scale|
| `analysis/Stereo_Analysis.py` | Post-processing for stereo PIV results: mean/RMS velocities, structure functions, dissipation rate, integral length scale, anisotropy invariants |

## Setup

```bash
pip install -r requirements.txt
```

Raw PIV data paths are currently hardcoded (e.g. `/Volumes/PIV Data1/...`) at the top of each script under a `# Control` section — update these to match your local data location before running.
