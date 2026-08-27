"""Standalone: mean flow / TKE contour + summary stats for stereo cases.

Loads only Averages.npz per case.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.discovery import discover_case_dirs_or_root
from common.prompts import ask_text

# --------------------------------------------
# Control
# --------------------------------------------
processed_root = input(
    "Enter the processed-results directory (Stereo_Decomposition.py output; "
    "a single case folder, or a parent of many): "
).strip()
only = ask_text("Limit to one case name (blank = process every case found)")

case_dirs = discover_case_dirs_or_root(
    processed_root, required_glob=os.path.join("Ensemble_Averages", "Averages.npz"))
if not case_dirs:
    raise FileNotFoundError(
        f"No Ensemble_Averages/Averages.npz found directly in {processed_root!r} "
        "or in its immediate subfolders.")
if only:
    case_dirs = {only: case_dirs[only]}

print(f"Found {len(case_dirs)} case(s): {', '.join(case_dirs)}")

for case_name, case_dir in case_dirs.items():
    print(f"\n===== {case_name} =====")

    avg_file = os.path.join(case_dir, "Ensemble_Averages", "Averages.npz")
    d = np.load(avg_file)
    X, Y, U_mean, V_mean, W_mean, U_rms, V_rms, W_rms, TKE, uv, uw, vw = (
        d["X"], d["Y"], d["U_mean"], d["V_mean"], d["W_mean"],
        d["U_rms"], d["V_rms"], d["W_rms"], d["TKE"], d["uv"], d["uw"], d["vw"])

    U_mean_avg = round(100 * np.nanmean(U_mean), 3)
    V_mean_avg = round(100 * np.nanmean(V_mean), 3)
    W_mean_avg = round(100 * np.nanmean(W_mean), 3)
    U_rms_avg = round(100 * np.nanmean(U_rms), 3)
    V_rms_avg = round(100 * np.nanmean(V_rms), 3)
    W_rms_avg = round(100 * np.nanmean(W_rms), 3)
    Isotropy_ratio_UV = np.nanmean(U_rms) / np.nanmean(V_rms)
    Isotropy_ratio_UW = np.nanmean(U_rms) / np.nanmean(W_rms)
    TKE_mean = np.nanmean(TKE) * 10000
    TKE_perdev = (np.nanstd(TKE) / np.nanmean(TKE)) * 100

    print(f'Mean U: {U_mean_avg} cm/s   Mean V: {V_mean_avg} cm/s   Mean W: {W_mean_avg} cm/s')
    print(f'RMS U: {U_rms_avg} cm/s   RMS V: {V_rms_avg} cm/s   RMS W: {W_rms_avg} cm/s')
    print(f'Isotropy ratio U/V: {Isotropy_ratio_UV}   U/W: {Isotropy_ratio_UW}')
    print(f'TKE_mean: {TKE_mean} cm^2/s^2   TKE % deviation: {TKE_perdev}')

    fig, ax = plt.subplots()
    ax.contourf(X, Y, gaussian_filter(TKE, 3), cmap='viridis', levels=10)
    ax.set_title(f'{case_name}: TKE')
    plt.show()
