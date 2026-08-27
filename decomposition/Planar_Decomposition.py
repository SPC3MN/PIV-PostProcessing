import os
import sys
import time
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.discovery import discover_case_dirs_or_root
from common.io_npz import load_dataset_npz
from common.prompts import ask_yes_no, ask_float, ask_int, ask_text
from common.decomposition_stats import (
    reynolds_decomp_planar,
    boot_ci,
    Structure_Function,
    Autocorrelation,
    Energy_Spectra,
)

warnings.filterwarnings("ignore", message="Mean of empty slice")

# --------------------------------------------
# Control
# --------------------------------------------
raw_root = input(
    "Enter the npz snapshot directory (a single case folder of snap_*.npz "
    "files, or a parent folder containing multiple case subfolders): "
).strip()

processed_root = ask_text("Enter the output directory for decomposition results")
if not processed_root:
    raise ValueError("An output directory is required.")

only = ask_text("Limit to one case name (blank = process every case found)")

n = ask_int("Limit snapshots per case (blank = no limit)")
cutoff_idx = n if n else False

if ask_yes_no("Crop each snapshot to a custom size before processing?", default=False):
    width = ask_float("  Crop width (mm)")
    height = ask_float("  Crop height (mm)")
    crop = (width, height)
else:
    crop = None

Auto = ask_yes_no("Calculate spatial autocorrelation?", default=True)
Structure = ask_yes_no("Calculate the structure function?", default=True)
Spectra = ask_yes_no("Calculate 1D energy spectra?", default=True)
Bootstrap = ask_yes_no("Compute bootstrap confidence intervals?", default=True)
Save_Fluct = ask_yes_no("Save fluctuating velocity snapshots?", default=False)
Save_NPZ = ask_yes_no("Re-save input snapshots into the output npz folder?", default=False)

case_dirs = discover_case_dirs_or_root(raw_root, required_glob="*.npz")
if not case_dirs:
    raise FileNotFoundError(
        f"No snap_*.npz files found directly in {raw_root!r} or in its immediate subfolders.")
if only:
    case_dirs = {only: case_dirs[only]}

print(f"Found {len(case_dirs)} case(s): {', '.join(case_dirs)}")

for case_name, input_dir in case_dirs.items():
    print(f"\n===== {case_name} =====")

    npz_dir = os.path.join(processed_root, case_name, "Full_npz")
    output_avg_dir = os.path.join(processed_root, case_name, "Ensemble_Averages")
    output_fluct_dir = os.path.join(processed_root, case_name, "Fluctuating_npz")
    code_start = time.perf_counter()

    # ── Load snapshots ─────────────────────────
    X, Y, U_all, V_all = load_dataset_npz(
        input_dir, cutoff_idx, components=("U", "V"), crop=crop)

    if Save_NPZ:
        print('Saving input as npz...')
        start = time.perf_counter()
        os.makedirs(npz_dir, exist_ok=True)

        for i in range(U_all.shape[0]):
            np.savez_compressed(
                os.path.join(npz_dir, f"snap_{i:03d}.npz"),
                X=X,
                Y=Y,
                U=U_all[i],
                V=V_all[i],
            )

        print(f"Saving done: {round(time.perf_counter() - start, 0)} s" + "\n")

    # ── Conduct phase decomposition ─────────────────────────
    start = time.perf_counter()
    U_mean, V_mean, U_fluct, V_fluct, U_rms, V_rms, TKE = reynolds_decomp_planar(U_all, V_all)

    print(f"Decomposition done: {round(time.perf_counter() - start, 0)} s" + "\n")

    # ── Save all phase averages ─────────────────────────
    os.makedirs(output_avg_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(output_avg_dir, f"Averages.npz"),
        X=X,
        Y=Y,
        U_mean=U_mean,
        V_mean=V_mean,
        U_rms=U_rms,
        V_rms=V_rms,
        TKE=TKE,
    )

    if Bootstrap:

        print('Bootstrapping mean velocity...')

        start = time.perf_counter()
        U_mean_lo, U_mean_hi = boot_ci(U_all.astype(np.float32))
        V_mean_lo, V_mean_hi = boot_ci(V_all.astype(np.float32))

        print(f'{round(time.perf_counter() - start, 0)} s')

        print('Bootstrapping RMS velocity...')
        start = time.perf_counter()
        U_rms_lo, U_rms_hi = boot_ci(U_fluct.astype(np.float32) ** 2)
        U_rms_lo, U_rms_hi = np.sqrt(U_rms_lo), np.sqrt(U_rms_hi)

        V_rms_lo, V_rms_hi = boot_ci(V_fluct.astype(np.float32) ** 2)
        V_rms_lo, V_rms_hi = np.sqrt(V_rms_lo), np.sqrt(V_rms_hi)

        print(f'{round(time.perf_counter() - start, 0)} s')

        np.savez_compressed(
            os.path.join(output_avg_dir, f"Bootstrapped_Statistics"),
            U_low=U_mean_lo,
            U_high=U_mean_hi,
            V_low=V_mean_lo,
            V_high=V_mean_hi,
            U_rms_low=U_rms_lo,
            U_rms_high=U_rms_hi,
            V_rms_low=V_rms_lo,
            V_rms_high=V_rms_hi,
        )

    if Structure:

        # ── Calculate structure function ───────────

        print('Calculating structure function...')
        start = time.perf_counter()
        D11, D33, D31, D13 = Structure_Function(U_fluct, V_fluct)

        np.savez_compressed(
            os.path.join(output_avg_dir, f"Structure_Function.npz"),
            X=X,
            Y=Y,
            D11=D11[0],
            D11_low=D11[1],
            D11_hi=D11[2],
            D33=D33[0],
            D33_low=D33[1],
            D33_hi=D33[2],
            D31=D31[0],
            D31_low=D31[1],
            D31_hi=D31[2],
            D13=D13[0],
            D13_low=D13[1],
            D13_hi=D13[2],
        )

        print("\n" + f"Structure functions done: {round(time.perf_counter() - start, 0)} s" + "\n")

    if Auto:

        # ── Calculate spatial autocorrelation ───────────

        print('Calculating spatial autocorrelation...')
        start = time.perf_counter()
        rho11, rho33, rho31, rho13 = Autocorrelation(U_fluct, V_fluct)

        np.savez_compressed(
            os.path.join(output_avg_dir, f"Autocorrelation_Function.npz"),
            X=X,
            Y=Y,
            rho11=rho11[0],
            rho11_low=rho11[1],
            rho11_hi=rho11[2],
            rho33=rho33[0],
            rho33_low=rho33[1],
            rho33_hi=rho33[2],
            rho31=rho31[0],
            rho31_low=rho31[1],
            rho31_hi=rho31[2],
            rho13=rho13[0],
            rho13_low=rho13[1],
            rho13_hi=rho13[2],
        )

        print("\n" + f"Autocorrelation functions done: {round(time.perf_counter() - start, 0)} s" + "\n")

    if Spectra:
        # ── Calculate 1D energy spectra ───────────

        print('Calculating 1D energy spectra...')
        start = time.perf_counter()
        dx = (abs(X[0][0] - X[0][1])) / 1000
        kx, Eu, ky, Ev = Energy_Spectra(U_fluct, V_fluct, dx)

        np.savez_compressed(
            os.path.join(output_avg_dir, f"Energy_Spectra.npz"),
            kx=kx,
            Eu=Eu,
            ky=ky,
            Ev=Ev
        )

        print("\n" + f"Energy spectra done: {round(time.perf_counter() - start, 0)} s" + "\n")

    # ── Save all fluctuating snapshots ─────────────────────────

    if Save_Fluct:
        print('Saving fluctuating snapshots...')
        os.makedirs(output_fluct_dir, exist_ok=True)

        for i in range(U_fluct.shape[0]):

            np.savez_compressed(
                os.path.join(output_fluct_dir, f"snap_{i:03d}.npz"),
                X=X,
                Y=Y,
                U_fluct=U_fluct[i],
                V_fluct=V_fluct[i]
            )

            print(f"\r{i + 1}/{len(U_fluct)}", end="")

    print("\n" + f"Total time for {case_name}: {round((time.perf_counter() - code_start)/60, 3)} min" + "\n")

    if Bootstrap:
        print(f'Mean U: {np.nanmean(U_mean)} ({np.nanmean(U_mean_lo)}, {np.nanmean(U_mean_hi)})')
        print(f'Mean V: {np.nanmean(V_mean)} ({np.nanmean(V_mean_lo)}, {np.nanmean(V_mean_hi)})')
        print(f'RMS U: {np.nanmean(U_rms)} ({np.nanmean(U_rms_lo)}, {np.nanmean(U_rms_hi)})')
        print(f'RMS V: {np.nanmean(V_rms)} ({np.nanmean(V_rms_lo)}, {np.nanmean(V_rms_hi)})')
        print(f'TKE: {np.nanmean((2*U_rms**2+V_rms**2)/2)} '
              f'({np.nanmean((2*U_rms_lo**2+V_rms_lo**2)/2)}, {np.nanmean((2*U_rms_hi**2+V_rms_hi**2)/2)})')
