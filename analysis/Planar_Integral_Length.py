"""Standalone: integral length scale (L_11, L_33) + Taylor microscale for
planar cases.

Loads Structure_Function.npz (only to locate the inertial-range fit window --
the same window Planar_Analysis.py reuses for both L_11 and L_33) and
Autocorrelation_Function.npz.
"""
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.discovery import discover_case_dirs_or_root
from common.prompts import ask_text
from common.analysis_stats import inertial_range_mask, taylor_microscale, fit_integral_length

warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

SLOPE_LO, SLOPE_HI = 0.6, 0.73  # inertial-range window (matches Planar_Analysis.py)
INTEGRAL_LENGTH_EXTEND = 5  # multiple of the fitted length scale to extrapolate the model past the data

# --------------------------------------------
# Control
# --------------------------------------------
processed_root = input(
    "Enter the processed-results directory (Planar_Decomposition.py output; "
    "a single case folder, or a parent of many): "
).strip()
only = ask_text("Limit to one case name (blank = process every case found)")

case_dirs = discover_case_dirs_or_root(
    processed_root, required_glob=os.path.join("Ensemble_Averages", "Autocorrelation_Function.npz"))
if not case_dirs:
    raise FileNotFoundError(
        f"No Ensemble_Averages/Autocorrelation_Function.npz found directly in {processed_root!r} "
        "or in its immediate subfolders.")
if only:
    case_dirs = {only: case_dirs[only]}

print(f"Found {len(case_dirs)} case(s): {', '.join(case_dirs)}")

for case_name, case_dir in case_dirs.items():
    print(f"\n===== {case_name} =====")

    structure_file = os.path.join(case_dir, "Ensemble_Averages", "Structure_Function.npz")
    auto_file = os.path.join(case_dir, "Ensemble_Averages", "Autocorrelation_Function.npz")

    ds = np.load(structure_file)
    conv = 10000  # m^2/s^2 to cm^2/s^2
    X = ds["X"]
    dr = (X[0, 1] - X[0, 0]) / 10  # mm to cm
    D_11 = ds["D11"] * conv
    D_33 = ds["D33"] * conv

    da = np.load(auto_file)
    rho_11 = da["rho11"]
    rho_11_low = da["rho11_low"]
    rho_11_hi = da["rho11_hi"]
    rho_33 = da["rho33"]
    rho_33_low = da["rho33_low"]
    rho_33_hi = da["rho33_hi"]

    r1_D = np.arange(0, len(D_11)) * dr
    r3_D = np.arange(0, len(D_33)) * dr
    max_beg11, max_end11 = inertial_range_mask(D_11, r1_D, SLOPE_LO, SLOPE_HI, drop_zero_lag=True)

    r1 = np.arange(0, len(rho_11)) * dr
    r3 = np.arange(0, len(rho_33)) * dr

    lambda_1 = taylor_microscale(r1, rho_11)
    lambda_3 = taylor_microscale(r3, rho_33)

    ######  L_11
    fit11 = fit_integral_length(
        rho_11, r1, max_beg11, max_end11, INTEGRAL_LENGTH_EXTEND,
        rho_low=rho_11_low, rho_hi=rho_11_hi)
    L_11 = fit11["L_ext"]

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

    axes[0].plot(fit11["r_trunc"], fit11["rho_trunc"], label=f'Data: L = {fit11["L_data"]}')
    axes[0].plot(fit11["r_trunc"], fit11["rho_low_trunc"], label='CI', linestyle='--')
    axes[0].plot(fit11["r_trunc"], fit11["rho_hi_trunc"], label='CI', linestyle='--')
    axes[0].plot(fit11["r_fit"], fit11["rho_fit"], 'r', label=f'Model fit: L = {fit11["L_fit"]}')
    axes[0].plot(fit11["r_ext"][len(fit11["r_fit"]):], fit11["rho_ext"][len(fit11["r_fit"]):], 'r',
                 label=f'Extended model: L = {fit11["L_ext"]}', linestyle='--')
    axes[0].set_ylabel(r'Autocorrelation Function $\rho_{ij}(r)$')
    axes[0].set_xlabel(r'r (mm)')
    axes[0].legend()

    ######  L_33  (note: reuses the D_11-derived mask window, matching Planar_Analysis.py)
    fit33 = fit_integral_length(
        rho_33, r3, max_beg11, max_end11, INTEGRAL_LENGTH_EXTEND,
        rho_low=rho_33_low, rho_hi=rho_33_hi)
    L_33 = fit33["L_ext"]

    axes[1].plot(fit33["r_trunc"], fit33["rho_trunc"], label=f'Data: L = {fit33["L_data"]}')
    axes[1].plot(fit33["r_fit"], fit33["rho_fit"], 'r', label=f'Model fit: L = {fit33["L_fit"]}')
    axes[1].plot(fit33["r_trunc"], fit33["rho_low_trunc"], label='CI', linestyle='--')
    axes[1].plot(fit33["r_trunc"], fit33["rho_hi_trunc"], label='CI', linestyle='--')
    axes[1].plot(fit33["r_ext"][len(fit33["r_fit"]):], fit33["rho_ext"][len(fit33["r_fit"]):], 'r',
                 label=f'Extended model: L = {fit33["L_ext"]}', linestyle='--')
    axes[1].set_xlabel(r'r (cm)')
    axes[1].legend()
    fig.suptitle(case_name)
    plt.show()

    print(f'L_11 = {L_11} mm   L_33 = {L_33} mm')
    print(f'lambda_1 (Taylor microscale) = {lambda_1}   lambda_3 = {lambda_3}')
