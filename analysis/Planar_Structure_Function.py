"""Standalone: second-order structure function D_11(r) plot for planar cases.

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
from common.analysis_stats import inertial_range_mask

warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

SLOPE_LO, SLOPE_HI = 0.6, 0.73  # inertial-range window: D(r) ~ r^slope target band (matches Planar_Analysis.py)

# --------------------------------------------
# Control
# --------------------------------------------
processed_root = input(
    "Enter the processed-results directory (Planar_Decomposition.py output; "
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
    dr = (X[0, 1] - X[0, 0]) / 10  # mm to cm

    D_11 = d["D11"] * conv
    D_11_low = d["D11_low"] * conv
    D_11_hi = d["D11_hi"] * conv
    D_33 = d["D33"] * conv

    r1 = np.arange(0, len(D_11)) * dr
    r3 = np.arange(0, len(D_33)) * dr

    max_beg11, max_end11 = inertial_range_mask(D_11, r1, SLOPE_LO, SLOPE_HI, drop_zero_lag=True)
    max_beg33, max_end33 = inertial_range_mask(D_33, r3, SLOPE_LO, SLOPE_HI, drop_zero_lag=True)

    plt.axvspan(r1[max_beg11], r1[max_end11], alpha=0.2, color='r', label='inertial range (D_11)')
    plt.axvspan(r3[max_beg33], r3[max_end33], alpha=0.2, color='b', label='inertial range (D_33)')

    plt.loglog(r1[1:], D_11[1:], label="D_11", c='r')
    plt.loglog(r1[1:], D_11_low[1:], label="CI", c='r', linestyle='--')
    plt.loglog(r1[1:], D_11_hi[1:], label="CI", c='r', linestyle='--')

    plt.xlabel(r'$r$ (cm)')
    plt.ylabel(r'cm^2/s^2')
    plt.title(f'{case_name}: Structure Function')
    plt.legend()
    plt.show()
