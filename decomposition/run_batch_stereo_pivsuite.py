"""
Stereo decomposition for cases processed through the custom PIV Suite
(piv_suite CLI, `{pair_id}_stereo_velocity.npz` raw output) rather than
DaVis/LaVision .vc7 export.

piv_suite's own raw output has three producer-specific quirks this loader
corrects at load time, BEFORE trim and crop -- confirmed against real DaVis
.vc7 data for recording "On Time=6.0_Burst On Time=0.0_Burst Off Time=0.0"
(corr(U)=+0.9885, corr(V)=-0.9904, corr(W)=+0.9904 over 245,865 overlapping
cells; error bound ~5mm, not sub-pixel -- see the investigation this was
built from):

  1. POSITION UNITS -- x/y are raw dewarped-world-canvas pixels, never
     scaled to mm anywhere in piv_suite (only velocity is, via
     world_scale_px_per_mm). The fix uses DaVis's own <LinearScaleX/Y>
     OffsetMm + FactorMmPerPixel straight from Calibration.xml -- the exact
     same slope piv_suite already decodes as world_scale_px_per_mm, plus
     the offset piv_suite parses and discards. y additionally needs
     un-flipping first: piv_suite's stereo path stores a row-up (display-
     flipped) y, while DaVis's LinearScaleY is defined against row-down
     canvas rows.
  2. V SIGN -- piv_suite's V is flipped relative to DaVis's convention
     (measured corr(V) = -0.9904 on the same overlap).
  3. INVALID-CELL FILL -- piv_suite's replace_invalid interpolates over
     invalid cells rather than leaving them NaN; valid=False cells are
     re-NaN'd here so only measured vectors enter the turbulence stats.

TRIM_PTS is 6, not the DaVis-derived pipeline's 8: piv_suite's correlation
grid already insets 2 points from DaVis's raw 384x735 (its final pass
leaves less margin), so 380x731 trimmed by 6/edge lands on the same
368x719 target shape as the DaVis path's 384x735 trimmed by 8/edge.

Order of operations, same convention as the other batches:
  1. Load raw snapshot -> convert x,y to mm, flip V, re-NaN invalid cells.
  2. TRIM 6 pts/edge.
  3. CROP (Crop10cm variant only) to a 10cm window about x=0, on the
     already-trimmed mm X -- reusing the same x_window_about_zero logic
     as the DaVis-derived stereo pipeline.

Output: J:\\PIV_PostProc\\Swirl_STEREO_TRIMMED\\<variant>\\<case>\\
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
from common.decomposition_stats import (
    reynolds_decomp_stereo,
    boot_ci,
    Structure_Function,
    Autocorrelation,
)
from common.anisotropy import reynolds_stress, compute_anisotropy_invariants

warnings.filterwarnings("ignore", message="Mean of empty slice")

RAW_ROOT = r"J:\PIV_PostProc\_npz_staging\CustomSuite"
OUT_ROOT = r"J:\PIV_PostProc\Swirl_STEREO_TRIMMED"

# J:\Final_Stereo\Properties\Calibration\Calibration.xml <Scales> block,
# verified byte-for-byte against this machine's own file before use.
SCALE_X_MM_PER_PX = 0.055800534820316119
OFFSET_X_MM = -156.63139402816938
SCALE_Y_MM_PER_PX = -0.055800534820316119
OFFSET_Y_MM = 77.405606872253855
CANVAS_NY = 3067  # CorrectedImageSize Height

TRIM_PTS = 6            # points removed from every edge (see module docstring)
CROP_WIDTH_MM = 100.0   # 10 cm window about x = 0, applied after the trim

VARIANTS = ("Stereo_FullFOV", "Stereo_Crop10cm")


def grid_mm(x_px, y_px):
    X = OFFSET_X_MM + SCALE_X_MM_PER_PX * x_px
    Y = OFFSET_Y_MM + SCALE_Y_MM_PER_PX * (CANVAS_NY - y_px)
    return X, Y


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


def load_case(input_dir, cutoff):
    """Load every snapshot, applying the mm conversion / V flip / re-NaN
    fixups, then trim TRIM_PTS points off every edge."""
    files = sorted(f for f in glob.glob(os.path.join(input_dir, '*.npz'))
                   if not os.path.basename(f).startswith('._'))
    if cutoff:
        files = files[:cutoff]

    with np.load(files[0]) as d0:
        X_full, Y_full = grid_mm(d0['x'], d0['y'])

    def _one(f):
        with np.load(f) as d:
            U = d['U'].astype(np.float32)
            V = -d['V'].astype(np.float32)          # producer V-sign flip
            W = d['W'].astype(np.float32)
            invalid = ~d['valid']
            U[invalid] = np.nan
            V[invalid] = np.nan
            W[invalid] = np.nan
        return U, V, W

    print(f'Loading {len(files)} files (mm conversion + trim {TRIM_PTS} pts/edge)... ', flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor() as ex:
        res = list(ex.map(_one, files))
    U = np.stack([r[0] for r in res])
    V = np.stack([r[1] for r in res])
    W = np.stack([r[2] for r in res])
    del res
    gc.collect()

    t = TRIM_PTS
    sl = (slice(t, -t), slice(t, -t))
    X = X_full[sl]
    Y = Y_full[sl]
    U = U[:, t:-t, t:-t]
    V = V[:, t:-t, t:-t]
    W = W[:, t:-t, t:-t]

    # piv_suite's row order runs Y-decreasing; the study's other snap_*.npz
    # sources run Y-increasing. Storage-order only -- flipping axis 0/1
    # together for every array keeps each row's X/Y/U/V/W correctly paired,
    # and no component's sign is touched by this (that's the separate V
    # flip above).
    X = X[::-1]
    Y = Y[::-1]
    U = U[:, ::-1]
    V = V[:, ::-1]
    W = W[:, ::-1]

    print(f'  -> {U.shape[1]} x {U.shape[2]} pts, '
          f'{U.shape[2]*abs(X[0,1]-X[0,0]):.1f} x {U.shape[1]*abs(Y[1,0]-Y[0,0]):.1f} mm '
          f'in {time.perf_counter()-t0:.1f} s', flush=True)
    return X, Y, U, V, W


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

    cases = {os.path.basename(d): d for d in sorted(glob.glob(os.path.join(RAW_ROOT, "*")))
              if os.path.isdir(d) and glob.glob(os.path.join(d, "*.npz"))}
    if args.only_case:
        cases = {k: v for k, v in cases.items() if k == args.only_case}
    variants = [args.variant] if args.variant else list(VARIANTS)

    print(f"{len(cases)} custom-suite case(s) x {len(variants)} variant(s); trim {TRIM_PTS} pts/edge")
    print(f"output root: {OUT_ROOT}\n")

    t_all = time.perf_counter()
    for n, (case_name, input_dir) in enumerate(sorted(cases.items()), 1):
        print(f"\n===== [{n}/{len(cases)}] {case_name} =====", flush=True)
        X, Y, U_all, V_all, W_all = load_case(input_dir, cutoff)

        if "Stereo_FullFOV" in variants:
            decompose(case_name, "Stereo_FullFOV", X, Y, U_all, V_all, W_all)

        if "Stereo_Crop10cm" in variants:
            l, r = x_window_about_zero(X, CROP_WIDTH_MM)
            decompose(case_name, "Stereo_Crop10cm", X[:, l:r], Y[:, l:r],
                      U_all[:, :, l:r], V_all[:, :, l:r], W_all[:, :, l:r])

        del X, Y, U_all, V_all, W_all
        gc.collect()

    print(f"\nALL DONE in {(time.perf_counter()-t_all)/60:.1f} min")
