"""
Re-runs decomposition for every planar case processed so far (23 cases under
J:\\PostProc_NPZ\\<recording>\\<case>\\ + 5 cases under D:\\Final_NPZ\\Swirl_PLANAR\\<case>\\),
this time cropped to a width of 10 cm (100 mm) centered on x=0 -- the
coordinate origin within the field, NOT the geometric center of the FOV.
Height is left at the full FOV extent (no y crop).

Output: Desktop/Swirl_Planar_Cropped/<case_name>/Ensemble_Averages/
"""
import os
import sys
import time
import warnings
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.discovery import discover_case_dirs_or_root
from common.io_npz import load_dataset_npz, _resolve_key
from common.decomposition_stats import (
    reynolds_decomp_planar,
    boot_ci,
    Structure_Function,
    Autocorrelation,
    Energy_Spectra,
)

warnings.filterwarnings("ignore", message="Mean of empty slice")

OUT_ROOT = os.path.join(os.path.expanduser("~"), "Desktop", "Swirl_Planar_Cropped")
CROP_WIDTH_MM = 100.0  # 10 cm, centered on x = 0 (not the FOV's geometric center)

RAW_ROOTS = [
    r"J:\PostProc_NPZ",       # folder-of-recordings-of-cases
    r"D:\Final_NPZ\Swirl_PLANAR",  # flat folder-of-cases
]


def _zero_centered_x_bounds(npz_dir, width_mm):
    """(top, bottom, left, right) bounds cropping only the x-extent to
    `width_mm` centered on x=0; y (top/bottom) spans the full FOV."""
    first = sorted(f for f in os.listdir(npz_dir) if f.endswith('.npz') and not f.startswith('._'))[0]
    with np.load(os.path.join(npz_dir, first)) as data:
        X_full = data[_resolve_key(data, 'X')]

    Ny0, Nx0 = X_full.shape
    x_coords = X_full[0]
    dx = np.median(np.diff(x_coords))

    zero_idx = int(np.argmin(np.abs(x_coords)))
    half = int(round((width_mm / 2) / abs(dx)))
    left = zero_idx - half
    right = zero_idx + half

    if left < 0 or right > Nx0:
        raise ValueError(
            f"{npz_dir}: {width_mm} mm centered at x=0 exceeds available FOV "
            f"(x range {x_coords.min():.1f} to {x_coords.max():.1f} mm, dx={dx:.4f} mm). "
            f"Computed indices: left={left}, right={right}, Nx0={Nx0}"
        )

    print(f'dx = {dx:.4f} mm; x=0 at column {zero_idx} (x={x_coords[zero_idx]:.3f} mm)')
    print(f'Cropping x to [{x_coords[left]:.2f}, {x_coords[right-1]:.2f}] mm '
          f'({right-left} points, {(right-left)*abs(dx):.2f} mm) -- height left uncropped ({Ny0} points)')

    return 0, Ny0, left, right


def discover_all_cases():
    """{case_name: case_dir} across every raw root, whether it's a flat folder
    of cases (root/<case>/*.npz) or a folder of recordings each containing
    case subfolders (root/<recording>/<case>/*.npz)."""
    cases = {}
    for root in RAW_ROOTS:
        found = discover_case_dirs_or_root(root, required_glob="*.npz")
        if not found:
            # root has neither *.npz directly nor case subfolders with *.npz --
            # assume it's a folder of recordings, each holding case subfolders
            for entry in sorted(os.listdir(root)):
                sub = os.path.join(root, entry)
                if os.path.isdir(sub):
                    found.update(discover_case_dirs_or_root(sub, required_glob="*.npz"))
        cases.update(found)
    return cases


def process_case(case_name, input_dir, output_avg_dir, cutoff_idx):
    code_start = time.perf_counter()

    bounds = _zero_centered_x_bounds(input_dir, CROP_WIDTH_MM)
    top, bottom, left, right = bounds

    npz_files = sorted(
        f for f in __import__('glob').glob(os.path.join(input_dir, '*.npz'))
        if not os.path.basename(f).startswith('._'))
    if cutoff_idx:
        npz_files = npz_files[:cutoff_idx]

    from concurrent.futures import ThreadPoolExecutor
    print(f'Loading {len(npz_files)} Files... ')

    def _load(f):
        print(f"\r{os.path.basename(f)}", end="")
        with np.load(f) as data:
            X = data[_resolve_key(data, 'X')][top:bottom, left:right]
            Y = data[_resolve_key(data, 'Y')][top:bottom, left:right]
            U = data[_resolve_key(data, 'U')][top:bottom, left:right]
            V = data[_resolve_key(data, 'V')][top:bottom, left:right]
        return X, Y, U, V

    t0 = time.perf_counter()
    with ThreadPoolExecutor() as ex:
        results = list(ex.map(_load, npz_files))
    X, Y = results[0][0], results[0][1]
    U_all = np.stack([r[2] for r in results])
    V_all = np.stack([r[3] for r in results])
    print(f"\nLoading done: {round(time.perf_counter() - t0, 3)} s\n")

    start = time.perf_counter()
    U_mean, V_mean, U_fluct, V_fluct, U_rms, V_rms, TKE = reynolds_decomp_planar(U_all, V_all)
    print(f"Decomposition done: {round(time.perf_counter() - start, 0)} s\n")

    os.makedirs(output_avg_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_avg_dir, "Averages.npz"),
        X=X, Y=Y, U_mean=U_mean, V_mean=V_mean, U_rms=U_rms, V_rms=V_rms, TKE=TKE,
    )

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

    case_dirs = discover_all_cases()
    if args.only_case:
        case_dirs = {k: v for k, v in case_dirs.items() if k == args.only_case}

    print(f"Found {len(case_dirs)} case(s): {', '.join(case_dirs)}")

    for case_name, input_dir in case_dirs.items():
        print(f"\n===== {case_name} =====")
        output_avg_dir = os.path.join(OUT_ROOT, case_name, "Ensemble_Averages")
        process_case(case_name, input_dir, output_avg_dir, cutoff)
