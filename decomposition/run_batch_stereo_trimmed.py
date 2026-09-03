"""
Stereo decomposition for all 7 swirl-stereo cases with the PIV edge artefacts
trimmed off, in two height-matched variants.

Order of operations (same convention as the planar batch):
  1. TRIM  - drop TRIM_PTS points from all four edges of the raw 384 x 735 field,
             removing the outermost interrogation windows.
             -> 368 x 719 pts, 163.8 x 320.5 mm  (6.25% of area removed)
  2. CROP  - only for the Stereo_Crop10cm variant, restrict x to a 10 cm window
             centred on x = 0 (the coordinate axis, not the centre of the FOV).

Trimming first means both variants keep exactly the same 368-point height.

Stereo differs from the planar pipeline: three components (U, V, W), Reynolds
stresses and Lumley anisotropy invariants are produced, and there is no energy
spectrum stage (the repo's Stereo_Decomposition.py does not compute one).
Structure functions and autocorrelations use U and V, matching that script.

Output: D:\\Final_NPZ\\Swirl_STEREO_TRIMMED\\<variant>\\<case>\\
"""
import os
import sys
import gc
import glob
import time
import warnings
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.discovery import discover_case_dirs_or_root
from common.io_npz import _resolve_key
from common.decomposition_stats import (
    reynolds_decomp_stereo,
    boot_ci,
    Structure_Function,
    Autocorrelation,
)
from common.anisotropy import reynolds_stress, compute_anisotropy_invariants

warnings.filterwarnings("ignore", message="Mean of empty slice")

RAW_ROOT = r"D:\Final_NPZ\Swirl_STEREO"
OUT_ROOT = r"D:\Final_NPZ\Swirl_STEREO_TRIMMED"

TRIM_PTS = 8           # points removed from every edge (6.25% of FOV area)
CROP_WIDTH_MM = 100.0  # 10 cm window about x = 0, applied after the trim

VARIANTS = ("Stereo_FullFOV", "Stereo_Crop10cm")


def load_trimmed(input_dir, cutoff):
    """Load every snapshot's U, V, W with TRIM_PTS points removed from all edges."""
    files = sorted(f for f in glob.glob(os.path.join(input_dir, '*.npz'))
                   if not os.path.basename(f).startswith('._'))
    if cutoff:
        files = files[:cutoff]
    t = TRIM_PTS
    sl = (slice(t, -t), slice(t, -t))

    with np.load(files[0]) as d0:
        X = d0[_resolve_key(d0, 'X')][sl]
        Y = d0[_resolve_key(d0, 'Y')][sl]

    def _one(f):
        with np.load(f) as d:
            return tuple(d[_resolve_key(d, c)][sl].astype(np.float32) for c in ("U", "V", "W"))

    print(f'Loading {len(files)} files (trimmed {t} pts/edge)... ', flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor() as ex:
        res = list(ex.map(_one, files))
    U = np.stack([r[0] for r in res])
    V = np.stack([r[1] for r in res])
    W = np.stack([r[2] for r in res])
    del res
    gc.collect()
    print(f'  -> {U.shape[1]} x {U.shape[2]} pts, '
          f'{U.shape[2]*abs(X[0,1]-X[0,0]):.1f} x {U.shape[1]*abs(Y[1,0]-Y[0,0]):.1f} mm '
          f'in {time.perf_counter()-t0:.1f} s', flush=True)
    return X, Y, U, V, W


def x_window_about_zero(X, width_mm):
    x = X[0]
    dx = np.median(np.diff(x))
    zero = int(np.argmin(np.abs(x)))
    half = int(round((width_mm / 2) / abs(dx)))
    left, right = zero - half, zero + half
    if left < 0 or right > len(x):
        raise ValueError(f"{width_mm} mm about x=0 does not fit in x[{x.min():.1f}, {x.max():.1f}]")
    print(f'  crop x -> [{x[left]:.2f}, {x[right-1]:.2f}] mm '
          f'({right-left} pts, {(right-left)*abs(dx):.2f} mm)', flush=True)
    return left, right


def decompose(case_name, variant, X, Y, U_all, V_all, W_all):
    base = os.path.join(OUT_ROOT, variant, case_name)
    avg_dir = os.path.join(base, "Ensemble_Averages")
    lum_dir = os.path.join(base, "Lumley_Statistics")
    os.makedirs(avg_dir, exist_ok=True)
    os.makedirs(lum_dir, exist_ok=True)
    t_case = time.perf_counter()
    print(f'-- {variant}: {U_all.shape[0]} snaps of {U_all.shape[1]} x {U_all.shape[2]}', flush=True)

    (U_mean, V_mean, W_mean, U_fluct, V_fluct, W_fluct,
     U_rms, V_rms, W_rms, TKE, uv, uw, vw) = reynolds_decomp_stereo(U_all, V_all, W_all)

    np.savez_compressed(os.path.join(avg_dir, "Averages.npz"),
                        X=X, Y=Y, U_mean=U_mean, V_mean=V_mean, W_mean=W_mean,
                        U_rms=U_rms, V_rms=V_rms, W_rms=W_rms, TKE=TKE,
                        uv=uv, uw=uw, vw=vw)

    print('Bootstrapping...', flush=True)
    t0 = time.perf_counter()
    U_lo, U_hi = boot_ci(U_all.astype(np.float32))
    V_lo, V_hi = boot_ci(V_all.astype(np.float32))
    W_lo, W_hi = boot_ci(W_all.astype(np.float32))
    Ur_lo, Ur_hi = boot_ci(U_fluct.astype(np.float32) ** 2)
    Ur_lo, Ur_hi = np.sqrt(Ur_lo), np.sqrt(Ur_hi)
    Vr_lo, Vr_hi = boot_ci(V_fluct.astype(np.float32) ** 2)
    Vr_lo, Vr_hi = np.sqrt(Vr_lo), np.sqrt(Vr_hi)
    Wr_lo, Wr_hi = boot_ci(W_fluct.astype(np.float32) ** 2)
    Wr_lo, Wr_hi = np.sqrt(Wr_lo), np.sqrt(Wr_hi)
    print(f'  {time.perf_counter()-t0:.0f} s', flush=True)
    np.savez_compressed(os.path.join(avg_dir, "Bootstrapped_Statistics"),
                        U_low=U_lo, U_high=U_hi, V_low=V_lo, V_high=V_hi,
                        W_low=W_lo, W_high=W_hi,
                        U_rms_low=Ur_lo, U_rms_high=Ur_hi,
                        V_rms_low=Vr_lo, V_rms_high=Vr_hi,
                        W_rms_low=Wr_lo, W_rms_high=Wr_hi)

    print('Structure function...', flush=True)
    t0 = time.perf_counter()
    D11, D33, D31, D13 = Structure_Function(U_fluct, V_fluct)
    np.savez_compressed(os.path.join(avg_dir, "Structure_Function.npz"), X=X, Y=Y,
                        D11=D11[0], D11_low=D11[1], D11_hi=D11[2],
                        D33=D33[0], D33_low=D33[1], D33_hi=D33[2],
                        D31=D31[0], D31_low=D31[1], D31_hi=D31[2],
                        D13=D13[0], D13_low=D13[1], D13_hi=D13[2])
    print(f'\n  {time.perf_counter()-t0:.0f} s', flush=True)

    print('Autocorrelation...', flush=True)
    t0 = time.perf_counter()
    r11, r33, r31, r13 = Autocorrelation(U_fluct, V_fluct)
    np.savez_compressed(os.path.join(avg_dir, "Autocorrelation_Function.npz"), X=X, Y=Y,
                        rho11=r11[0], rho11_low=r11[1], rho11_hi=r11[2],
                        rho33=r33[0], rho33_low=r33[1], rho33_hi=r33[2],
                        rho31=r31[0], rho31_low=r31[1], rho31_hi=r31[2],
                        rho13=r13[0], rho13_low=r13[1], rho13_hi=r13[2])
    print(f'\n  {time.perf_counter()-t0:.0f} s', flush=True)

    print('Lumley anisotropy...', flush=True)
    R, k = reynolds_stress(U_rms, V_rms, W_rms, uv, uw, vw)
    eta2, xi, II, III = compute_anisotropy_invariants(R, k)
    np.savez_compressed(os.path.join(lum_dir, "Lumley_Statistics.npz"),
                        R=R, eta2=eta2, xi=xi, II=II, III=III)

    print(f'   {variant} / {case_name} done in {(time.perf_counter()-t_case)/60:.2f} min '
          f'| RMS U/V/W {np.nanmean(U_rms):.4f}/{np.nanmean(V_rms):.4f}/{np.nanmean(W_rms):.4f} '
          f'| TKE {np.nanmean(TKE)*10000:.1f}\n', flush=True)

    del (U_mean, V_mean, W_mean, U_fluct, V_fluct, W_fluct,
         U_rms, V_rms, W_rms, TKE, uv, uw, vw, R)
    gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=0, help="limit snapshots per case (0 = all)")
    ap.add_argument("--only-case", type=str, default=None)
    ap.add_argument("--variant", choices=list(VARIANTS), default=None)
    args = ap.parse_args()
    cutoff = args.cutoff if args.cutoff > 0 else False

    cases = discover_case_dirs_or_root(RAW_ROOT, required_glob="*.npz")
    if args.only_case:
        cases = {k: v for k, v in cases.items() if k == args.only_case}
    variants = [args.variant] if args.variant else list(VARIANTS)

    print(f"{len(cases)} stereo case(s) x {len(variants)} variant(s); trim {TRIM_PTS} pts/edge")
    print(f"output root: {OUT_ROOT}\n")

    t_all = time.perf_counter()
    for n, (case_name, input_dir) in enumerate(sorted(cases.items()), 1):
        print(f"\n===== [{n}/{len(cases)}] {case_name} =====", flush=True)
        X, Y, U_all, V_all, W_all = load_trimmed(input_dir, cutoff)

        if "Stereo_FullFOV" in variants:
            decompose(case_name, "Stereo_FullFOV", X, Y, U_all, V_all, W_all)

        if "Stereo_Crop10cm" in variants:
            l, r = x_window_about_zero(X, CROP_WIDTH_MM)
            decompose(case_name, "Stereo_Crop10cm", X[:, l:r], Y[:, l:r],
                      U_all[:, :, l:r], V_all[:, :, l:r], W_all[:, :, l:r])

        del X, Y, U_all, V_all, W_all
        gc.collect()

    print(f"\nALL DONE in {(time.perf_counter()-t_all)/60:.1f} min")
