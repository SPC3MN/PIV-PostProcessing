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
| `decomposition/Stereo_Decomposition.py` | Reynolds/triple decomposition for stereo PIV data |
| `decomposition/Test_Decomposition.py` | Development version of the stereo decomposition pipeline |
| `analysis/Planar_Analysis.py` | Post-processing/analysis for planar PIV results |
| `analysis/Stereo_Analysis.py` | Post-processing: structure functions, dissipation rate, integral length scale, Lumley triangle |
| `analysis/Anisotropy_Invariants.py` | Reynolds stress anisotropy tensor and Lumley invariants |
| `analysis/Statistics_Plotting.py` | TKE and anisotropy invariant plotting from averaged fields |
| `analysis/Bayesian_Bootstrap.py` | Vectorized Dirichlet/Bayesian bootstrap confidence intervals |
| `plotting/Plotting.py` | Plots swept-parameter results (burst timing, TKE, IR, etc.) from CSV |
| `plotting/Clean_Plotting.py` | Publication-formatted comparison plots across cases |
| `jet_control/Sunbathing.py` | Generates pump control signals (standard + pulsed "sunbathing" burst modes) |
| `single_jet/Single_Jet.py` | Single-jet field statistics: spreading rate, KE, enstrophy, velocity profiles |
| `single_jet/Test.py` | Development/exploration version of the single-jet analysis |

## Setup

```bash
pip install -r requirements.txt
```

Raw PIV data paths are currently hardcoded (e.g. `/Volumes/PIV Data1/...`) at the top of each script under a `# Control` section — update these to match your local data location before running.

## Notes

- `analysis/Anisotropy_Invariants.py`, `analysis/Statistics_Plotting.py`, and `decomposition/Test_Decomposition.py` each carry their own copies of `reynolds_stress()` / `compute_anisotropy_invariants()` — worth consolidating into a shared module at some point.
- `single_jet/Test.py` is largely a duplicate of `single_jet/Single_Jet.py`; kept separate here since both were in the uploaded set, but consider merging or removing.
