import os
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import warnings


warnings.filterwarnings("ignore", message="Mean of empty slice")


def load_single_npz(file, top, bottom, left, right):
    print(f"\r{os.path.basename(file)}", end = "")
    with np.load(file) as data:
        return data['U'][top:bottom, left:right], data['V'][top:bottom, left:right]


def load_dataset(npz_dir, cutoff, width_mm, height_mm):
    """Load snapshots previously exported by this pipeline's Save_NPZ step
    (snap_*.npz files holding X, Y, U, V), cropping each to a centered
    width_mm x height_mm window."""
    npz_files = sorted(
        f for f in glob.glob(os.path.join(npz_dir, '*.npz')) if not os.path.basename(f).startswith('._'))
    if cutoff:
        npz_files = npz_files[:cutoff]

    print(f'Loading {len(npz_files)} Files... ')

    with np.load(npz_files[0]) as data0:
        X_full, Y_full = data0['X'], data0['Y']

    Ny0, Nx0 = X_full.shape
    print(f'Original Size: {(Ny0, Nx0)}')

    x_coords_full = X_full[0]
    y_coords_full = Y_full[:, 0]

    dx = np.median(np.diff(x_coords_full))
    dy = np.median(np.diff(y_coords_full))
    print(f'dx = {dx:.4f} mm, dy = {dy:.4f} mm')

    x_center = (x_coords_full.min() + x_coords_full.max()) / 2
    y_center = (y_coords_full.min() + y_coords_full.max()) / 2
    print(f'FOV center: x={x_center:.3f} mm, y={y_center:.3f} mm')

    Nx = int(round(width_mm / abs(dx)))
    Ny = int(round(height_mm / abs(dy)))

    x_center_idx = np.argmin(np.abs(x_coords_full - x_center))
    y_center_idx = np.argmin(np.abs(y_coords_full - y_center))

    left = x_center_idx - Nx // 2
    right = left + Nx
    top = y_center_idx - Ny // 2
    bottom = top + Ny

    if left < 0 or top < 0 or right > Nx0 or bottom > Ny0:
        raise ValueError(
            f"Requested frame ({width_mm} x {height_mm} mm) exceeds available FOV "
            f"({Nx0*abs(dx):.1f} x {Ny0*abs(dy):.1f} mm). "
            f"Computed indices: left={left}, right={right}, top={top}, bottom={bottom}"
        )

    print(f'Trimmed to: {(Ny, Nx)} points -> {Nx*abs(dx):.2f} x {Ny*abs(dy):.2f} mm, '
          f'centered at ({x_center:.2f}, {y_center:.2f}) mm')

    X = X_full[top:bottom, left:right]
    Y = Y_full[top:bottom, left:right]

    with ThreadPoolExecutor() as ex:
        results = list(ex.map(lambda f: load_single_npz(f, top, bottom, left, right), npz_files))

    # filter out empty results (any array with shape (0, 0))
    results = [
        r for r in results
        if all(arr.shape != (0, 0) for arr in r)
    ]

    U_all, V_all = zip(*results)

    print("\n" + f"Loading done: {round(time.perf_counter() - start, 3)} s" + "\n")

    return X, Y, np.stack(U_all), np.stack(V_all)
def reynolds_decomp(U_all, V_all):


    print('Calculating Mean... ')
    U_mean = np.nanmean(U_all, axis=0)   # (Ny, Nx)
    V_mean = np.nanmean(V_all, axis=0)

    print('Subtracting mean from each snapshot... ')
    U_fluct = U_all - U_mean             # (N, Ny, Nx)
    V_fluct = V_all - V_mean

    print('Calculating RMS... ')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Degrees of freedom', category=RuntimeWarning)
        uu = np.nanvar(U_all, axis=0)        # same as nanmean(fluct**2)
        vv = np.nanvar(V_all, axis=0)

    U_rms = np.sqrt(uu)
    V_rms = np.sqrt(vv)
    TKE   = (2*uu + vv) / 2          # (Ny, Nx)

    return U_mean, V_mean, U_fluct, V_fluct, U_rms, V_rms, TKE
def boot_ci(data, B=500, alpha=0.10, chunk=5000, downsample=1):
    """
    data  : (Nt, Ny, Nx) float32

    Returns
    -------
    lo, hi : (Ny, Nx) float32, NaN where all-NaN input
    """
    data = data[:, ::downsample, ::downsample]
    Nt = data.shape[0]
    spatial_shape = data.shape[1:]
    flat = data.reshape(Nt, -1)  # (Nt, Ny*Nx)

    # Mask: pixels where any snapshot is NaN
    nan_mask = np.isnan(flat).any(axis=0)         # (Ny*Nx,)
    flat_clean = flat.copy()
    flat_clean[:, nan_mask] = 0.0                  # zero-fill so matmul stays fast

    n_spatial = flat_clean.shape[1]

    W_boot = np.random.dirichlet(np.ones(Nt), size=B).astype(np.float32)

    lo = np.empty(n_spatial, dtype=np.float32)
    hi = np.empty(n_spatial, dtype=np.float32)

    lo_q = alpha / 2
    hi_q = 1 - alpha / 2

    for start in range(0, n_spatial, chunk):
        sl = slice(start, min(start + chunk, n_spatial))
        b_chunk = W_boot @ flat_clean[:, sl]
        lo[sl] = np.quantile(b_chunk, lo_q, axis=0)
        hi[sl] = np.quantile(b_chunk, hi_q, axis=0)

    # Restore NaN at masked pixels
    lo[nan_mask] = np.nan
    hi[nan_mask] = np.nan

    return lo.reshape(spatial_shape), hi.reshape(spatial_shape)
def boot_ci_autocorrelation(velocity_field, B=500, alpha=0.10):
    """
    Dirichlet bootstrap CI for the normalized two-point autocorrelation.

    Parameters
    ----------
    velocity_field : (Ny, Nx) float32
    B              : number of bootstrap resamples
    alpha          : two-tailed significance level

    Returns
    -------
    corr    : (Nx,) point estimate
    lo, hi  : (Nx,) float32 confidence interval
    """
    ny, nx = velocity_field.shape

    valid = (~np.isnan(velocity_field)).astype(np.float32)        # (Ny, Nx)
    field = np.where(np.isnan(velocity_field), 0.0, velocity_field).astype(np.float32)

    n = 2 * nx

    F       = np.fft.rfft(field, n=n, axis=1)    # (Ny, n//2+1)
    F_valid = np.fft.rfft(valid, n=n, axis=1)

    # Per-row autocorrelations — one (Nx,) array per row
    corr_rows  = np.fft.irfft(F       * np.conj(F),       n=n, axis=1)[:, :nx]  # (Ny, Nx)
    count_rows = np.fft.irfft(F_valid * np.conj(F_valid), n=n, axis=1)[:, :nx]  # (Ny, Nx)

    # Dirichlet weights : (B, Ny)
    W = np.random.dirichlet(np.ones(ny), size=B).astype(np.float32)

    # Weighted sum over rows for each bootstrap draw
    boot_corr  = W @ corr_rows.astype(np.float32)   # (B, Nx)
    boot_count = W @ count_rows.astype(np.float32)  # (B, Nx)

    # Normalised autocorrelation for each draw
    boot_norm  = boot_corr / boot_count.clip(1e-8)  # (B, Nx)
    boot_norm /= boot_norm[:, [0]]                  # divide by lag-0 per draw

    lo = np.quantile(boot_norm, alpha / 2,     axis=0).astype(np.float32)
    hi = np.quantile(boot_norm, 1 - alpha / 2, axis=0).astype(np.float32)

    # Point estimate
    corr_sum  = corr_rows.sum(axis=0)
    count_sum = count_rows.sum(axis=0)
    corr_mean = corr_sum / count_sum.clip(1)
    corr      = (corr_mean / corr_mean[0]).astype(np.float32)

    return corr, lo, hi
def boot_ci_structure_function(velocity_field, B=500, alpha=0.10):
    """
    Dirichlet bootstrap CI for the second-order structure function.

    Parameters
    ----------
    velocity_field : (Ny, Nx) float32
    B              : number of bootstrap resamples
    alpha          : two-tailed significance level

    Returns
    -------
    D       : (Nx,) point estimate
    lo, hi  : (Nx,) float32 confidence interval
    """
    ny, nx = velocity_field.shape

    field = np.where(np.isnan(velocity_field), 0.0, velocity_field).astype(np.float32)
    count = (~np.isnan(velocity_field)).astype(np.float32)

    n = 2 * nx

    F  = np.fft.rfft(field,      n=n, axis=1)
    F2 = np.fft.rfft(field ** 2, n=n, axis=1)
    Fc = np.fft.rfft(count,      n=n, axis=1)

    # Per-row contributions : (Ny, Nx) each
    cross_sq_rows = np.fft.irfft(F2 * np.conj(Fc) + Fc * np.conj(F2), n=n, axis=1)[:, :nx]
    auto_rows     = np.fft.irfft(F  * np.conj(F),                      n=n, axis=1)[:, :nx]
    n_pairs_rows  = np.fft.irfft(Fc * np.conj(Fc),                     n=n, axis=1)[:, :nx]

    # Dirichlet weights : (B, Ny)
    W = np.random.dirichlet(np.ones(ny), size=B).astype(np.float32)

    # Weighted sums over rows for each bootstrap draw
    boot_cross_sq = W @ cross_sq_rows   # (B, Nx)
    boot_auto     = W @ auto_rows       # (B, Nx)
    boot_n_pairs  = W @ n_pairs_rows    # (B, Nx)

    # Structure function per draw : D(r) = (cross_sq - 2*auto) / n_pairs
    boot_D = (boot_cross_sq - 2 * boot_auto) / boot_n_pairs.clip(1e-8)   # (B, Nx)

    lo = np.quantile(boot_D, alpha / 2,     axis=0).astype(np.float32)
    hi = np.quantile(boot_D, 1 - alpha / 2, axis=0).astype(np.float32)

    # Point estimate
    D = ((cross_sq_rows.sum(axis=0) - 2 * auto_rows.sum(axis=0))
         / n_pairs_rows.sum(axis=0).clip(1)).astype(np.float32)

    return D, lo, hi
def Autocorrelation(U_fluct, V_fluct):
    rho11 = np.empty(len(U_fluct), dtype=object)
    rho11_low = np.empty(len(U_fluct), dtype=object)
    rho11_hi = np.empty(len(U_fluct), dtype=object)
    rho33 = np.empty(len(U_fluct), dtype=object)
    rho33_low = np.empty(len(U_fluct), dtype=object)
    rho33_hi = np.empty(len(U_fluct), dtype=object)
    rho13 = np.empty(len(U_fluct), dtype=object)
    rho13_low = np.empty(len(U_fluct), dtype=object)
    rho13_hi = np.empty(len(U_fluct), dtype=object)
    rho31 = np.empty(len(U_fluct), dtype=object)
    rho31_low = np.empty(len(U_fluct), dtype=object)
    rho31_hi = np.empty(len(U_fluct), dtype=object)

    for i, (U_snap, V_snap) in enumerate(zip(U_fluct, V_fluct)):
        rho11[i], rho11_low[i], rho11_hi[i] = boot_ci_autocorrelation(U_snap)
        rho33[i], rho33_low[i], rho33_hi[i] = boot_ci_autocorrelation(V_snap.T)
        rho13[i], rho13_low[i], rho13_hi[i] = boot_ci_autocorrelation(U_snap.T)
        rho31[i], rho31_low[i], rho31_hi[i] = boot_ci_autocorrelation(V_snap)

        print(f"\r{i + 1}/{len(U_fluct)}", end="")

    rho11 = [np.mean(rho11, axis=0), np.mean(rho11_low, axis=0), np.mean(rho11_hi, axis=0)]

    rho33 = [np.mean(rho33, axis=0), np.mean(rho33_low, axis=0), np.mean(rho33_hi, axis=0)]

    rho31 = np.mean(rho31, axis=0), np.mean(rho31_low, axis=0), np.mean(rho31_hi, axis=0)

    rho13 = np.mean(rho13, axis=0), np.mean(rho13_low, axis=0), np.mean(rho13_hi, axis=0)


    return rho11, rho33, rho31, rho13
def Structure_Function(U_fluct, V_fluct):
    D11 = np.empty(len(U_fluct), dtype=object)
    D11_low = np.empty(len(U_fluct), dtype=object)
    D11_hi = np.empty(len(U_fluct), dtype=object)
    D33 = np.empty(len(U_fluct), dtype=object)
    D33_low = np.empty(len(U_fluct), dtype=object)
    D33_hi = np.empty(len(U_fluct), dtype=object)
    D13 = np.empty(len(U_fluct), dtype=object)
    D13_low = np.empty(len(U_fluct), dtype=object)
    D13_hi = np.empty(len(U_fluct), dtype=object)
    D31 = np.empty(len(U_fluct), dtype=object)
    D31_low = np.empty(len(U_fluct), dtype=object)
    D31_hi = np.empty(len(U_fluct), dtype=object)

    for i, (U_snap, V_snap) in enumerate(zip(U_fluct, V_fluct)):
        D11[i], D11_low[i], D11_hi[i] = boot_ci_structure_function(U_snap)
        D33[i], D33_low[i], D33_hi[i] = boot_ci_structure_function(V_snap.T)
        D13[i], D13_low[i], D13_hi[i] = boot_ci_structure_function(U_snap.T)
        D31[i], D31_low[i], D31_hi[i] = boot_ci_structure_function(V_snap)

        print(f"\r{i + 1}/{len(U_fluct)}", end="")

    D11 = [np.mean(D11, axis=0), np.mean(D11_low, axis=0), np.mean(D11_hi, axis=0)]

    D33 = [np.mean(D33, axis=0), np.mean(D33_low, axis=0), np.mean(D33_hi, axis=0)]

    D31 = np.mean(D31, axis=0), np.mean(D31_low, axis=0), np.mean(D31_hi, axis=0)

    D13 = np.mean(D13, axis=0), np.mean(D13_low, axis=0), np.mean(D13_hi, axis=0)

    return D11, D33, D31, D13
def spatial_spectrum_1d(fluct_field, dx):
    """
    fluct_field : (Ny, Nx) fluctuating velocity component
    dx          : grid spacing [m]
    Returns k (rad/m), E(k) ensemble-averaged over rows
    """
    ny, nx = fluct_field.shape
    valid = ~np.isnan(fluct_field).any(axis=1)
    field = fluct_field[valid]

    # window to reduce spectral leakage (finite FOV isn't periodic)
    win = np.hanning(nx)
    win_correction = np.mean(win**2)
    field_win = field * win[None, :]

    F = np.fft.rfft(field_win, axis=1)
    psd = (np.abs(F)**2) / (nx * win_correction)   # per-row PSD
    psd_mean = psd.mean(axis=0)                     # average over rows

    k = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)
    return k[1:], psd_mean[1:]
def Energy_Spectra(U_fluct, V_fluct, dx):

    Eu = np.empty(len(U_fluct), dtype=object)
    Ev = np.empty(len(V_fluct), dtype=object)

    kx = None
    ky = None

    for i, (U_snap, V_snap) in enumerate(zip(U_fluct, V_fluct)):

        kx_i, Eu[i] = spatial_spectrum_1d(U_snap, dx)
        ky_i, Ev[i] = spatial_spectrum_1d(V_snap.T, dx)

        if kx is None:
            kx = kx_i
        if ky is None:
            ky = ky_i

        print(f"\r{i+1}/{len(U_fluct)}", end="")

    Eu = np.mean(Eu, axis=0)
    Ev = np.mean(Ev, axis=0)

    return kx, Eu, ky, Ev

# --------------------------------------------
# Control
# --------------------------------------------
width, height = 200, 150  # mm, size of the centered crop window applied to each snapshot
cutoff_idx = False
Auto = True
Structure = True
Spectra = True
Save_Fluct = False
Save_NPZ = False
Bootstrap = True

input_dir = input("Enter the case directory")
cases = [input("Enter the case or cases: ")]

for case in cases:

    input_dir = input_dir + case
    npz_dir = 'npz_dir' + case + '/Full_npz'
    output_avg_dir = 'output_dir' + case + '/Ensemble_Averages'
    output_fluct_dir = 'output_dir2' + case + '/Fluctuating_npz'
    output_boot_dir = 'output_dr3' + case + '/Bootstrap'
    code_start = time.perf_counter()

    # ── Load snapshots ─────────────────────────
    start = time.perf_counter()
    X, Y, U_all, V_all = load_dataset(input_dir, cutoff_idx, width, height)

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
    U_mean, V_mean, U_fluct, V_fluct, U_rms, V_rms, TKE = reynolds_decomp(U_all, V_all)

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

        start = time.perf_counter()

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

        # ── Calculate spatial autocorrelation ───────────

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
        # ── Calculate spatial autocorrelation ───────────

        print('Calculating 1D energy spectra...')
        start = time.perf_counter()
        dx = (abs(X[0][0] - X[0][1]))/1000
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
        start = time.perf_counter()
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



    print("\n" + f"Total time: {round((time.perf_counter() - code_start)/60, 3)} min" + "\n")


    print(f'Mean U: {np.nanmean(U_mean)} ({np.nanmean(U_mean_lo)}, {np.nanmean(U_mean_hi)})')
    print(f'Mean V: {np.nanmean(V_mean)} ({np.nanmean(V_mean_lo)}, {np.nanmean(V_mean_hi)})')
    print(f'RMS U: {np.nanmean(U_rms)} ({np.nanmean(U_rms_lo)}, {np.nanmean(U_rms_hi)})')
    print(f'RMS V: {np.nanmean(V_rms)} ({np.nanmean(V_rms_lo)}, {np.nanmean(V_rms_hi)})')
    print(f'TKE: {np.nanmean((2*U_rms**2+V_rms**2)/2)} ({np.nanmean((2*U_rms_lo**2+V_rms_lo**2)/2)}, {np.nanmean((2*U_rms_hi**2+V_rms_hi**2)/2)})')
