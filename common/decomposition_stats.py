import warnings
import numpy as np


def reynolds_decomp_planar(U_all, V_all):
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
    TKE   = (2 * uu + vv) / 2          # (Ny, Nx)

    return U_mean, V_mean, U_fluct, V_fluct, U_rms, V_rms, TKE


def reynolds_decomp_stereo(U_all, V_all, W_all):
    print('Calculating Mean... ')
    U_mean = np.nanmean(U_all, axis=0)   # (Ny, Nx)
    V_mean = np.nanmean(V_all, axis=0)
    W_mean = np.nanmean(W_all, axis=0)

    print('Subtracting mean from each snapshot... ')
    U_fluct = U_all - U_mean             # (N, Ny, Nx)
    V_fluct = V_all - V_mean
    W_fluct = W_all - W_mean

    print('Calculating RMS... ')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Degrees of freedom', category=RuntimeWarning)
        uu = np.nanvar(U_all, axis=0)        # same as nanmean(fluct**2)
        vv = np.nanvar(V_all, axis=0)
        ww = np.nanvar(W_all, axis=0)

    U_rms = np.sqrt(uu)
    V_rms = np.sqrt(vv)
    W_rms = np.sqrt(ww)
    TKE   = (uu + vv + ww) / 2          # (Ny, Nx)

    print('Calculating off diagonal ... ')
    uv = np.nanmean(U_fluct * V_fluct, axis=0)
    uw = np.nanmean(U_fluct * W_fluct, axis=0)
    vw = np.nanmean(V_fluct * W_fluct, axis=0)

    return U_mean, V_mean, W_mean, U_fluct, V_fluct, W_fluct, U_rms, V_rms, W_rms, TKE, uv, uw, vw


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
