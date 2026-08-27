"""Standalone: compensated structure function / dissipation-rate estimate
(eps_11, eps_33) for stereo cases.

Loads only Structure_Function.npz per case.
"""
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.discovery import discover_case_dirs_or_root
from common.prompts import ask_text
from common.analysis_stats import inertial_range_mask, dissipation_rate_array

warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")

SLOPE_LO, SLOPE_HI = 0.54, 0.7  # inertial-range window (matches Stereo_Analysis.py)
SMOOTH_SIGMA = 3
C2 = 2

# --------------------------------------------
# Control
# --------------------------------------------
processed_root = input(
    "Enter the processed-results directory (Stereo_Decomposition.py output; "
    "a single case folder, or a parent of many): "
).strip()
only = ask_text("Limit to one case name (blank = process every case found)")

case_dirs = discover_case_dirs_or_root(
    processed_root, required_glob=os.path.join("Ensemble_Averages", "Structure_Function.npz"))
if not case_dirs:
    raise FileNotFoundError(
        f"No Ensemble_Averages/Structure_Function.npz found directly in {processed_root!r} "
        "or in its immediate subfolders.")
if only:
    case_dirs = {only: case_dirs[only]}

print(f"Found {len(case_dirs)} case(s): {', '.join(case_dirs)}")

for case_name, case_dir in case_dirs.items():
    print(f"\n===== {case_name} =====")

    structure_file = os.path.join(case_dir, "Ensemble_Averages", "Structure_Function.npz")
    d = np.load(structure_file)

    conv = 10000  # m^2/s^2 to cm^2/s^2
    X = d["X"]
    dr = (X[0, 1] - X[0, 0]) / 10

    D_11 = d["D11"][1:-10] * conv
    D_11_low = d["D11_low"][1:-10] * conv
    D_11_hi = d["D11_hi"][1:-10] * conv
    D_33 = d["D33"][1:-10] * conv
    D_33_low = d["D33_low"][1:-10] * conv
    D_33_hi = d["D33_hi"][1:-10] * conv

    r1 = np.arange(0, len(D_11)) * dr
    r3 = np.arange(0, len(D_33)) * dr

    max_beg11, max_end11 = inertial_range_mask(D_11, r1, SLOPE_LO, SLOPE_HI, smooth_sigma=SMOOTH_SIGMA)
    max_beg33, max_end33 = inertial_range_mask(D_33, r3, SLOPE_LO, SLOPE_HI, smooth_sigma=SMOOTH_SIGMA)

    eps_11 = dissipation_rate_array(D_11, r1, C2)
    eps_11_low = dissipation_rate_array(D_11_low, r1, C2)
    eps_11_hi = dissipation_rate_array(D_11_hi, r1, C2)

    eps_33 = dissipation_rate_array(D_33, r3, C2)
    eps_33_low = dissipation_rate_array(D_33_low, r3, C2)
    eps_33_hi = dissipation_rate_array(D_33_hi, r3, C2)

    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)

    axes[0].semilogx(r1, eps_11, label='e_11', c='r')
    axes[0].semilogx(r1, eps_11_low, label='CI', c='black', linestyle='--')
    axes[0].semilogx(r1, eps_11_hi, label='CI', c='black', linestyle='--')

    axes[1].semilogx(r3, eps_33, label='e_33', c='b')
    axes[1].semilogx(r3, eps_33_low, label='CI', c='black', linestyle='--')
    axes[1].semilogx(r3, eps_33_hi, label='CI', c='black', linestyle='--')

    axes[0].axhline(np.mean(eps_11[max_beg11:max_end11]), color='r', linestyle='--', label=f'eps = {np.mean(eps_11[max_beg11:max_end11])}')
    axes[1].axhline(np.mean(eps_33[max_beg33:max_end33]), color='b', linestyle='--', label=f'eps = {np.mean(eps_33[max_beg33:max_end33])}')

    axes[0].set_xlabel(r'$r$ (cm)')
    axes[0].set_ylabel(r'$cm^2/s^3$')
    axes[1].set_xlabel(r'$r$ (cm)')
    axes[0].set_title(f'{case_name}')
    axes[0].legend()
    axes[1].legend()
    plt.show()

    eps_11_mean = np.mean(eps_11[max_beg11:max_end11])
    eps_33_mean = np.mean(eps_33[max_beg33:max_end33])
    print(f'eps_11 = {eps_11_mean} cm^2/s^3')
    print(f'eps_33 = {eps_33_mean} cm^2/s^3')
