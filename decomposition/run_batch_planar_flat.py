"""
Same as run_batch_planar.py, but for a RAW_ROOT that is itself a flat folder
of case subfolders (no recording-date grouping level in between) -- i.e.
RAW_ROOT/<case_name>/snap_*.npz directly, rather than
RAW_ROOT/<recording_name>/<case_name>/snap_*.npz.

Output goes to OUT_ROOT/<case_name>_PROC/Ensemble_Averages/, matching the
"_PROC" suffix convention applied to the existing case folders already in
OUT_ROOT.
"""
import os
import sys
import time
import warnings
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.discovery import discover_case_dirs_or_root
from common.io_npz import load_dataset_npz
from common.decomposition_stats import (
    reynolds_decomp_planar,
    boot_ci,
    Structure_Function,
    Autocorrelation,
    Energy_Spectra,
)

warnings.filterwarnings("ignore", message="Mean of empty slice")

RAW_ROOT = r"D:\Final_NPZ\Swirl_PLANAR"
OUT_ROOT = r"D:\Final_NPZ\Swirl_PLANAR_PROCESSED"

crop = None  # no cropping beyond whatever padding the npz export already applied
Auto = True
Structure = True
Spectra = True
Bootstrap = True
Save_Fluct = False
Save_NPZ = False


def process_case(case_name, input_dir, output_avg_dir, cutoff_idx):
    code_start = time.perf_counter()

    X, Y, U_all, V_all = load_dataset_npz(input_dir, cutoff_idx, components=("U", "V"), crop=crop)

    start = time.perf_counter()
    U_mean, V_mean, U_fluct, V_fluct, U_rms, V_rms, TKE = reynolds_decomp_planar(U_all, V_all)
    print(f"Decomposition done: {round(time.perf_counter() - start, 0)} s\n")

    os.makedirs(output_avg_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_avg_dir, "Averages.npz"),
        X=X, Y=Y, U_mean=U_mean, V_mean=V_mean, U_rms=U_rms, V_rms=V_rms, TKE=TKE,
    )

    U_mean_lo = U_mean_hi = V_mean_lo = V_mean_hi = None
    U_rms_lo = U_rms_hi = V_rms_lo = V_rms_hi = None

    if Bootstrap:
        print('Bootstrapping mean velocity...')
        t0 = time.perf_counter()
        U_mean_lo, U_mean_hi = boot_ci(U_all.astype(np.float32))
        V_mean_lo, V_mean_hi = boot_ci(V_all.astype(np.float32))
        print(f'{round(time.perf_counter() - t0, 0)} s')

        print('Bootstrapping RMS velocity...')
        t0 = time.perf_counter()
        U_rms_lo, U_rms_hi = boot_ci(U_fluct.astype(np.float32) ** 2)
        U_rms_lo, U_rms_hi = np.sqrt(U_rms_lo), np.sqrt(U_rms_hi)
        V_rms_lo, V_rms_hi = boot_ci(V_fluct.astype(np.float32) ** 2)
        V_rms_lo, V_rms_hi = np.sqrt(V_rms_lo), np.sqrt(V_rms_hi)
        print(f'{round(time.perf_counter() - t0, 0)} s')

        np.savez_compressed(
            os.path.join(output_avg_dir, "Bootstrapped_Statistics"),
            U_low=U_mean_lo, U_high=U_mean_hi, V_low=V_mean_lo, V_high=V_mean_hi,
            U_rms_low=U_rms_lo, U_rms_high=U_rms_hi, V_rms_low=V_rms_lo, V_rms_high=V_rms_hi,
        )

    if Structure:
        print('Calculating structure function...')
        t0 = time.perf_counter()
        D11, D33, D31, D13 = Structure_Function(U_fluct, V_fluct)
        np.savez_compressed(
            os.path.join(output_avg_dir, "Structure_Function.npz"),
            X=X, Y=Y,
            D11=D11[0], D11_low=D11[1], D11_hi=D11[2],
            D33=D33[0], D33_low=D33[1], D33_hi=D33[2],
            D31=D31[0], D31_low=D31[1], D31_hi=D31[2],
            D13=D13[0], D13_low=D13[1], D13_hi=D13[2],
        )
        print(f"\nStructure functions done: {round(time.perf_counter() - t0, 0)} s\n")

    if Auto:
        print('Calculating spatial autocorrelation...')
        t0 = time.perf_counter()
        rho11, rho33, rho31, rho13 = Autocorrelation(U_fluct, V_fluct)
        np.savez_compressed(
            os.path.join(output_avg_dir, "Autocorrelation_Function.npz"),
            X=X, Y=Y,
            rho11=rho11[0], rho11_low=rho11[1], rho11_hi=rho11[2],
            rho33=rho33[0], rho33_low=rho33[1], rho33_hi=rho33[2],
            rho31=rho31[0], rho31_low=rho31[1], rho31_hi=rho31[2],
            rho13=rho13[0], rho13_low=rho13[1], rho13_hi=rho13[2],
        )
        print(f"\nAutocorrelation functions done: {round(time.perf_counter() - t0, 0)} s\n")

    if Spectra:
        print('Calculating 1D energy spectra...')
        t0 = time.perf_counter()
        dx = abs(X[0][0] - X[0][1]) / 1000
        kx, Eu, ky, Ev = Energy_Spectra(U_fluct, V_fluct, dx)
        np.savez_compressed(
            os.path.join(output_avg_dir, "Energy_Spectra.npz"),
            kx=kx, Eu=Eu, ky=ky, Ev=Ev,
        )
        print(f"\nEnergy spectra done: {round(time.perf_counter() - t0, 0)} s\n")

    print(f"\nTotal time for {case_name}: {round((time.perf_counter() - code_start) / 60, 3)} min\n")
    if Bootstrap:
        print(f'Mean U: {np.nanmean(U_mean)} ({np.nanmean(U_mean_lo)}, {np.nanmean(U_mean_hi)})')
        print(f'Mean V: {np.nanmean(V_mean)} ({np.nanmean(V_mean_lo)}, {np.nanmean(V_mean_hi)})')
        print(f'RMS U: {np.nanmean(U_rms)} ({np.nanmean(U_rms_lo)}, {np.nanmean(U_rms_hi)})')
        print(f'RMS V: {np.nanmean(V_rms)} ({np.nanmean(V_rms_lo)}, {np.nanmean(V_rms_hi)})')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=0, help="limit snapshots per case (0 = all)")
    ap.add_argument("--only-case", type=str, default=None, help="process only this case name")
    args = ap.parse_args()

    cutoff = args.cutoff if args.cutoff > 0 else False

    case_dirs = discover_case_dirs_or_root(RAW_ROOT, required_glob="*.npz")
    if args.only_case:
        case_dirs = {k: v for k, v in case_dirs.items() if k == args.only_case}

    print(f"Found {len(case_dirs)} case(s): {', '.join(case_dirs)}")

    for case_name, input_dir in case_dirs.items():
        print(f"\n===== {case_name} =====")
        output_avg_dir = os.path.join(OUT_ROOT, f"{case_name}_PROC", "Ensemble_Averages")
        process_case(case_name, input_dir, output_avg_dir, cutoff)
