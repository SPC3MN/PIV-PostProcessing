import os
import glob
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", message="Mean of empty slice")


def load_single(file, Ny, Nx):
    print(f"\r{os.path.basename(file)}", end = "")
    df = pd.read_csv(file, sep=';', usecols=['x [mm]', 'y [mm]', 'Velocity u [m/s]', 'Velocity v [m/s]', 'Velocity w [m/s]'])
    U = df.pivot_table(index='y [mm]', columns='x [mm]', values='Velocity u [m/s]').values
    V = df.pivot_table(index='y [mm]', columns='x [mm]', values='Velocity v [m/s]').values
    W = df.pivot_table(index='y [mm]', columns='x [mm]', values='Velocity w [m/s]').values

    return U[:Ny, :Nx], V[:Ny, :Nx], W[:Ny, :Nx]


def load_single_npz(file):
    print(f"\r{os.path.basename(file)}", end = "")
    with np.load(file) as data:
        return data['U'], data['V'], data['W']


def load_dataset_npz(npz_dir, cutoff):
    """Load snapshots previously exported by this pipeline's Save_NPZ step
    (snap_*.npz files holding X, Y, U, V, W)."""
    npz_files = sorted(
        f for f in glob.glob(os.path.join(npz_dir, '*.npz')) if not os.path.basename(f).startswith('._'))
    if cutoff:
        npz_files = npz_files[:cutoff]

    print(f'Loading {len(npz_files)} Files... ')

    with np.load(npz_files[0]) as data0:
        X, Y = data0['X'], data0['Y']

    with ThreadPoolExecutor() as ex:
        results = list(ex.map(load_single_npz, npz_files))

    U_all, V_all, W_all = zip(*results)

    print("\n" + f"Loading done: {round(time.perf_counter() - start, 3)} s" + "\n")

    return X, Y, np.stack(U_all), np.stack(V_all), np.stack(W_all)


def load_dataset(csv_dir, cutoff, input_format='auto'):

    if input_format == 'auto':
        has_npz = bool(glob.glob(os.path.join(csv_dir, '*.npz')))
        has_csv = bool(glob.glob(os.path.join(csv_dir, '*.csv')))
        input_format = 'npz' if has_npz and not has_csv else 'csv'

    if input_format == 'npz':
        return load_dataset_npz(csv_dir, cutoff)
    elif input_format != 'csv':
        raise ValueError(f"Unknown input_format: {input_format!r} (use 'csv', 'npz', or 'auto')")

    if cutoff:
        csv_files = sorted(
            f for f in glob.glob(os.path.join(csv_dir, '*.csv')) if not os.path.basename(f).startswith('._'))[:cutoff]

    else:
        csv_files = sorted(
            f for f in glob.glob(os.path.join(csv_dir, '*.csv')) if not os.path.basename(f).startswith('._'))

    print(f'Loading {len(csv_files)} Files... ')
    df0 = pd.read_csv(csv_files[1], sep=';')
    U0 = df0.pivot_table(index='y [mm]', columns='x [mm]', values='Velocity u [m/s]')
    Ny, Nx = U0.shape

    # round down to the nearest 10th
    Ny = (Ny // 10) * 10
    Nx = (Nx // 10) * 10

    x_coords = U0.columns.values
    y_coords = U0.index.values
    X, Y = np.meshgrid(x_coords, y_coords)
    X = X[:Ny, :Nx]
    Y = Y[:Ny, :Nx]

    with ThreadPoolExecutor() as ex:
        results = list(ex.map(lambda f: load_single(f, Ny, Nx), csv_files))

    # filter out empty results (any array with shape (0, 0))
    results = [
        r for r in results
        if all(arr.shape != (0, 0) for arr in r)
    ]

    U_all, V_all, W_all = zip(*results)

    print("\n" + f"Loading done: {round(time.perf_counter() - start, 3)} s" + "\n")

    return  X, Y, np.stack(U_all), np.stack(V_all), np.stack(W_all)

def reynolds_decomp(U_all, V_all, W_all):


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

def reynolds_stress(U_rms, V_rms, W_rms, uv, uw, vw):
    """
    Parameters
    ----------
    U_rms, V_rms, W_rms       : np.ndarray (Ny, Nx)
    uv, uw, vw                : np.ndarray (N, Ny, Nx)

    Returns
    -------
    R : np.ndarray (3, 3, Ny, Nx)
        Reynolds stress tensor at each grid point
    """

    # Diagonal terms from rms
    uu = U_rms**2
    vv = V_rms**2
    ww = W_rms**2

    # Assemble symmetric tensor (3, 3, Ny, Nx)
    R = np.array([
        [uu,  uv,  uw],
        [uv,  vv,  vw],
        [uw,  vw,  ww]
    ])

    k = 0.5 * (uu + vv + ww)

    return R, k

def compute_anisotropy_invariants(R, k):
    """
    Compute the anisotropy tensor bᵢⱼ and its invariants II and III.

    bᵢⱼ = ⟨u'ᵢu'ⱼ⟩ / (2k) - δᵢⱼ / 3
    II  = bᵢⱼ bⱼᵢ         (trace of b²)
    III = bᵢⱼ bⱼₖ bₖᵢ     (trace of b³)

    Returns eta² = II/3  and  ξ = (III/2)^(1/3)  [signed cube root]
    following Lumley & Newman (1977).

    Parameters
    ----------
    R : np.ndarray (3, 3, Ny, Nx)
        Reynolds stress tensor
    k : np.ndarray (Ny, Nx)
        Turbulent kinetic energy

    Returns
    -------
    eta2, xi, II, III : np.ndarray (Ny, Nx)
    """

    # Identity matrix
    I3 = np.eye(3)

    # Guard against zero TKE
    k_safe = np.where(k < 1e-15, np.nan, k)

    # Computes the anisotropy tensor bij= (ui′ * uj′)/2k − δij/3 (reshape I3 so it broadcasts correctly)
    b = R / (2.0 * k_safe) - I3[:, :, np.newaxis, np.newaxis] / 3.0

    # Move spatial dimensionss first for einsum: (Ny, Nx, 3, 3)
    b = np.moveaxis(b, [0, 1], [-2, -1])

    # Matrix multiplication at each grid point. b2 is b_ij * b_jk and b3 is b_ij * b_jk * b_kl
    b2 = b @ b        # (Ny, Nx, 3, 3)
    b3 = b2 @ b       # (Ny, Nx, 3, 3)

    # Takes the trace of the (3,3) matrix at each grid point, giving the invariants II = b_ij * b_ji and III = b_ij * b_jk * b_ki
    II  = np.trace(b2, axis1=-2, axis2=-1)   # (Ny, Nx)
    III = np.trace(b3, axis1=-2, axis2=-1)   # (Ny, Nx)

    # Converts to the Lumley-Newman coordinates. eta2: eta^2 = II/3 and xi: xi = (III/2)^1/3
    eta2 = II
    xi   = np.sign(III) * np.abs(III) ** (1.0 / 3.0)

    return eta2, xi, II, III

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

# --------------------------------------------
# Control
# --------------------------------------------
cutoff_idx = False # False if none
input_format = 'auto'  # 'csv', 'npz' (previously-exported snap_*.npz), or 'auto' to detect from input_dir
Auto = True
Structure = True
Save_Fluct = False
Save_NPZ = False
Save_Lumley = True
Bootstrap = True
case = '6s-12_5p'

# case = input("Enter the case name: ")

input_dir = '/Volumes/PIV Data1/Final_Stereo/6s-12_5p'
npz_dir = '/Volumes/PIV Data1/Stereo_Processed/' + case + '/Full_npz'
output_avg_dir = '/Volumes/PIV Data1/Stereo_Processed/' + case + '/Ensemble_Averages'
output_Lum_dir = '/Volumes/PIV Data1/Stereo_Processed/' + case + '/Lumley_Statistics'
output_fluct_dir = '/Volumes/PIV Data1/Stereo_Processed/' + case + '/Fluctuating_npz'
output_boot_dir = '/Volumes/PIV Data1/Stereo_Processed/' + case + '/Bootstrap'
code_start = time.perf_counter()


# ── Load snapshots ─────────────────────────
start = time.perf_counter()
X, Y, U_all, V_all, W_all = load_dataset(input_dir, cutoff=cutoff_idx, input_format=input_format)

dr = (X[0, 1] - X[0, 0])/10

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
            W=W_all[i]
        )

    print(f"Saving done: {round(time.perf_counter() - start, 0)} s" + "\n")

# ── Conduct phase decomposition ─────────────────────────
start = time.perf_counter()
U_mean, V_mean, W_mean, U_fluct, V_fluct, W_fluct,U_rms, V_rms, W_rms, TKE, uv, uw, vw = reynolds_decomp(U_all, V_all, W_all)

print(f"Decomposition done: {round(time.perf_counter() - start, 0)} s" + "\n")

# ── Save all phase averages ─────────────────────────
os.makedirs(output_avg_dir, exist_ok=True)

np.savez_compressed(
    os.path.join(output_avg_dir, f"Averages.npz"),
    X=X,
    Y=Y,
    U_mean=U_mean,
    V_mean=V_mean,
    W_mean=W_mean,
    U_rms=U_rms,
    V_rms=V_rms,
    W_rms=W_rms,
    TKE=TKE,
    uv=uv,
    uw=uw,
    vw=vw
)

if Bootstrap:

    start = time.perf_counter()

    print('Bootstrapping mean velocity...')

    start = time.perf_counter()
    U_mean_lo, U_mean_hi = boot_ci(U_all.astype(np.float32))
    V_mean_lo, V_mean_hi = boot_ci(V_all.astype(np.float32))
    W_mean_lo, W_mean_hi = boot_ci(W_all.astype(np.float32))
    print(f'{round(time.perf_counter() - start, 0)} s')

    print('Bootstrapping RMS velocity...')
    start = time.perf_counter()
    U_rms_lo, U_rms_hi = boot_ci(U_fluct.astype(np.float32) ** 2)
    U_rms_lo, U_rms_hi = np.sqrt(U_rms_lo), np.sqrt(U_rms_hi)

    V_rms_lo, V_rms_hi = boot_ci(V_fluct.astype(np.float32) ** 2)
    V_rms_lo, V_rms_hi = np.sqrt(V_rms_lo), np.sqrt(V_rms_hi)

    W_rms_lo, W_rms_hi = boot_ci(W_fluct.astype(np.float32) ** 2)
    W_rms_lo, W_rms_hi = np.sqrt(W_rms_lo), np.sqrt(W_rms_hi)

    print(f'{round(time.perf_counter() - start, 0)} s')

    np.savez_compressed(
        os.path.join(output_avg_dir, f"Bootstrapped_Statistics"),
        U_low=U_mean_lo,
        U_high=U_mean_hi,
        V_low=V_mean_lo,
        V_high=V_mean_hi,
        W_low=W_mean_lo,
        W_high=W_mean_hi,
        U_rms_low=U_rms_lo,
        U_rms_high=U_rms_hi,
        V_rms_low=V_rms_lo,
        V_rms_high=V_rms_hi,
        W_rms_low=W_rms_lo,
        W_rms_high=W_rms_hi,
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


if Save_Lumley:
    # ── Calculate Lumley Statistics ───────────
    os.makedirs(output_Lum_dir, exist_ok=True)

    print('Generating reynolds stress tensor... ')
    start = time.perf_counter()
    R, k = reynolds_stress(U_rms, V_rms, W_rms, uv, uw, vw)

    print('Calculating anisotropy invariants... ')
    eta2, xi, II, III = compute_anisotropy_invariants(R, k)

    np.savez_compressed(
        os.path.join(output_Lum_dir, f"Lumley_Statistics.npz"),
        R=R,
        eta2=eta2,
        xi=xi,
        II=II,
        III=III
    )

    print("\n" + f"Lumley statistics done: {round(time.perf_counter() - start, 0)} s" + "\n")


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
            V_fluct=V_fluct[i],
            W_fluct=W_fluct[i]
        )

        print(f"\r{i + 1}/{len(U_fluct)}", end="")



print("\n" + f"Total time: {round((time.perf_counter() - code_start)/60, 3)} min" + "\n")


print(f'Mean U: {np.nanmean(U_mean)} ({np.nanmean(U_mean_lo)}, {np.nanmean(U_mean_hi)})')
print(f'Mean V: {np.nanmean(V_mean)} ({np.nanmean(V_mean_lo)}, {np.nanmean(V_mean_hi)})')
print(f'Mean W: {np.nanmean(W_mean)} ({np.nanmean(W_mean_lo)}, {np.nanmean(W_mean_hi)})')
print(f'RMS U: {np.nanmean(U_rms)} ({np.nanmean(U_rms_lo)}, {np.nanmean(U_rms_hi)})')
print(f'RMS V: {np.nanmean(V_rms)} ({np.nanmean(V_rms_lo)}, {np.nanmean(V_rms_hi)})')
print(f'RMS W: {np.nanmean(W_rms)} ({np.nanmean(W_rms_lo)}, {np.nanmean(W_rms_hi)})')
print(f'TKE: {np.nanmean((U_rms**2+V_rms**2+W_rms**2)/2)} ({np.nanmean((U_rms_lo**2+V_rms_lo**2+W_rms_lo**2)/2)}, {np.nanmean((U_rms_hi**2+V_rms_hi**2+W_rms_hi**2)/2)})')