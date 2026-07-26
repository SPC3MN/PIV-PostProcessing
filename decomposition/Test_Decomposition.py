import os
import glob
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import warnings

warnings.filterwarnings("ignore", message="Mean of empty slice")


def load_single(file, Ny, Nx):
    print(f"\r{os.path.basename(file)}", end = "")
    df = pd.read_csv(file, sep=';', usecols=['x [mm]', 'y [mm]', 'Velocity u [m/s]', 'Velocity v [m/s]', 'Velocity w [m/s]'])
    U = df.pivot_table(index='y [mm]', columns='x [mm]', values='Velocity u [m/s]').values
    V = df.pivot_table(index='y [mm]', columns='x [mm]', values='Velocity v [m/s]').values
    W = df.pivot_table(index='y [mm]', columns='x [mm]', values='Velocity w [m/s]').values

    return U[:Ny, :Nx], V[:Ny, :Nx], W[:Ny, :Nx]

def load_dataset(csv_dir, cutoff):

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

def reynolds_decomp(U_all, V_all, W_all, B=1000, alpha=0.10):
    """
    Compute decomposition of velocity fields.

    Parameters
    ----------
    U_all, V_all, W_all : np.ndarray
        Shape (N, Ny, Nx)

    Returns
    -------
    U_mean, V_mean, W_mean : np.ndarray
        Global average
        Shape (Ny, Nx)

    U_fluct, V_fluct, W_fluct : np.ndarray
        Fluctuations
        Shape: (N, Ny, Nx)

    U_rms, V_rms, W_rms : np.ndarray
        RMS of the fluctuations
        Shape: (Ny, Nx)

    UV, UW, VW : np.ndarray
        Off diagonal Reynolds stress
        Shape: (Ny, Nx)

    """

    Nt = U_all.shape[0]


    UVW   = np.stack([U_all, V_all, W_all])             # (3, Nt, Ny, Nx)

    print('Calculating Mean... ')
    mean  = np.nanmean(UVW, axis=1)                     # (3, Ny, Nx)

    print('Subtracting mean from each snapshot... ')
    fluct = UVW - mean[:, np.newaxis]                   # (3, Nt, Ny, Nx)

    u, v, w = fluct  # each (Nt, Ny, Nx)

    print('Calculating RMS... ')
    # --- point estimates ---
    rms_sq = np.nanmean(fluct**2, axis=1)               # (3, Ny, Nx)
    rms    = np.sqrt(rms_sq)
    TKE    = rms_sq.sum(axis=0) / 2

    print('Calculating off diagonal... ')
    uv = np.nanmean(u * v, axis=0)
    uw = np.nanmean(u * w, axis=0)
    vw = np.nanmean(v * w, axis=0)


    print('Bootstrapping...')
    U_mean_lo, U_mean_hi = boot_ci(UVW[0])         # (Nt, Ny, Nx)
    V_mean_lo, V_mean_hi = boot_ci(UVW[1])
    W_mean_lo, W_mean_hi = boot_ci(UVW[2])
    rms_lo, rms_hi = boot_ci(np.sqrt(fluct ** 2))
    TKE_lo, TKE_hi = boot_ci((fluct**2).sum(axis=0) / 2)   # (Ny, Nx)

    U_mean, V_mean, W_mean = mean
    U_fluct, V_fluct, W_fluct = fluct
    U_rms, V_rms, W_rms = rms

    CI = dict(
        U = (U_mean_lo, U_mean_hi),
        V = (V_mean_lo, V_mean_hi),
        W = (W_mean_lo, W_mean_hi),
        rms = (rms_lo, rms_hi),   # each (3, Ny, Nx)
        TKE = (TKE_lo, TKE_hi),
    )

    return U_mean, V_mean, W_mean, U_fluct, V_fluct, W_fluct, U_rms, V_rms, W_rms, TKE, uv, uw, vw, CI

def Structure_Function(velocity_field):
    """
        Computes the second-order structure function D^2(r) for a 2D velocity field.

        The structure function measures the mean squared velocity difference between
        points separated by a lag distance r:

            D(r) = < [u(x + r) - u(x)]^2 >

        where the average is taken over all positions x and all rows.

        Parameters
        ----------
        velocity_field : np.ndarray, shape (ny, nx)
            2D array where each row is a 1D spatial snapshot of the velocity component.

        Returns
        -------
        D_ij : np.ndarray, shape (nx,)
            Structure function values for lags r = 0, 1, ..., nx-1.
            D_ij[0] is always 0 (zero separation = zero difference).
            D_ij[r] is the mean squared velocity increment at lag r.
        """
    ny, nx = velocity_field.shape
    D_ij = np.full(nx, np.nan)

    for r in range(nx):
        u1 = velocity_field[:, r:]
        u2 = velocity_field[:, :nx - r]

        diffs_sq = (u1 - u2) ** 2

        D_ij[r] = np.nanmean(diffs_sq)

    return D_ij

def Normalized_TwoPoint_Autocorrelation(velocity_field):
    """
        Computes the normalized two-point autocorrelation of a 2D velocity field.

        For each lag r, the autocorrelation is computed as the sum of products
        v[i] * v[i+r] across all rows and valid column pairs, then normalized
        by the zero-lag value R(0).

        Parameters
        ----------
        velocity_field : np.ndarray, shape (ny, nx)
            2D array where each row is a spatial velocity profile.
            NaN values are ignored in the summation.

        Returns
        -------
        corr : np.ndarray, shape (nx,)
            Normalized autocorrelation for lags 0 to nx-1.
            corr[0] = 1 by definition.
        """
    ny, nx = velocity_field.shape

    corr_sum = np.zeros(nx)
    for r in range(nx):
        a, b = velocity_field[:, :nx - r], velocity_field[:, r:]
        corr_sum[r] = np.nansum(a * b)

    return corr_sum / corr_sum[0]

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


# --------------------------------------------
# Control
# --------------------------------------------
cutoff_idx = 15 # False if none
Save_Fluct = True
Bootstrap = True
case = '6-12_5'


input_dir = '/Volumes/PIV Data1/' + case
npz_dir = '/Volumes/PIV Data1/Stereo_Processed/' + case + '/Full_npz'
output_avg_dir = '/Volumes/PIV Data1/Stereo_Processed/' + case + '/Ensemble_Averages'
output_Lum_dir = '/Volumes/PIV Data1/Stereo_Processed/' + case + '/Lumley_Statistics'
output_fluct_dir = '/Volumes/PIV Data1/Stereo_Processed/' + case + '/Fluctuating_npz'
code_start = time.perf_counter()


# ── Load snapshots ─────────────────────────
start = time.perf_counter()
X, Y, U_all, V_all, W_all = load_dataset(input_dir, cutoff=cutoff_idx)

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

print(f"Saving done: {round(time.perf_counter() - start, 3)} s" + "\n")

# ── Conduct phase decomposition ─────────────────────────
start = time.perf_counter()
U_mean, V_mean, W_mean, U_fluct, V_fluct, W_fluct, U_rms, V_rms, W_rms, TKE, uv, uw, vw, CI = (reynolds_decomp(U_all, V_all, W_all))

print(f"Decomposition done: {round(time.perf_counter() - start, 3)} s" + "\n")

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

# ── Calculate structure function ───────────

print('Calculating structure function...')
start = time.perf_counter()
D_11, D_13, D_33, D_31, D_21, D_23 = [], [], [], [], [], []

for i, (U_snap, V_snap, W_snap) in enumerate(zip(U_fluct, V_fluct, W_fluct)):

    D_11.append(Structure_Function(U_snap))
    D_13.append(Structure_Function(U_snap.T))
    D_33.append(Structure_Function(V_snap.T))
    D_31.append(Structure_Function(V_snap))
    D_21.append(Structure_Function(W_snap))
    D_23.append(Structure_Function(W_snap.T))

    print(f"\r{i + 1}/{len(U_fluct)}", end="")

D_11_mean = np.nanmean(D_11, axis=0)
D_33_mean = np.nanmean(D_33, axis=0)
D_13_mean = np.nanmean(D_13, axis=0)
D_31_mean = np.nanmean(D_31, axis=0)
D_21_mean = np.nanmean(D_21, axis=0)
D_23_mean = np.nanmean(D_23, axis=0)

np.savez_compressed(
    os.path.join(output_avg_dir, f"Structure_Function.npz"),
    X=X,
    Y=Y,
    D_11=D_11_mean,
    D_13=D_13_mean,
    D_33=D_33_mean,
    D_31=D_31_mean,
    D_21=D_21_mean,
    D_23=D_23_mean,
)

print("\n" + f"Structure functions done: {round(time.perf_counter() - start, 3)} s" + "\n")


# ── Calculate spatial autocorrelation ───────────

print('Calculating spatial autocorrelation...')
start = time.perf_counter()
rho_11, rho_13, rho_33, rho_31, rho_21, rho_23 = [], [], [], [], [], []

for i, (U_snap, V_snap, W_snap) in enumerate(zip(U_fluct, V_fluct, W_fluct)):
    rho_11.append(Normalized_TwoPoint_Autocorrelation(U_snap))
    rho_13.append(Normalized_TwoPoint_Autocorrelation(U_snap.T))
    rho_33.append(Normalized_TwoPoint_Autocorrelation(V_snap.T))
    rho_31.append(Normalized_TwoPoint_Autocorrelation(V_snap))
    rho_21.append(Normalized_TwoPoint_Autocorrelation(W_snap))
    rho_23.append(Normalized_TwoPoint_Autocorrelation(W_snap.T))

    print(f"\r{i + 1}/{len(U_fluct)}", end="")

rho_11_mean = np.nanmean(rho_11, axis=0)
rho_33_mean = np.nanmean(rho_33, axis=0)
rho_13_mean = np.nanmean(rho_13, axis=0)
rho_31_mean = np.nanmean(rho_31, axis=0)
rho_21_mean = np.nanmean(rho_21, axis=0)
rho_23_mean = np.nanmean(rho_23, axis=0)

np.savez_compressed(
    os.path.join(output_avg_dir, f"Autocorrelation_Function.npz"),
    X=X,
    Y=Y,
    rho_11=rho_11_mean,
    rho_13=rho_13_mean,
    rho_33=rho_33_mean,
    rho_31=rho_31_mean,
    rho_21=rho_21_mean,
    rho_23=rho_23_mean,
)

print("\n" + f"Autocorrelation functions done: {round(time.perf_counter() - start, 3)} s" + "\n")


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

print("\n" + f"Lumley statistics done: {round(time.perf_counter() - start, 3)} s" + "\n")


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


print("\n" + f"Total time: {(time.perf_counter() - code_start)//60} min" + "\n")


if Bootstrap:

    def boot_ci(snapshot_field, B=1000, alpha=0.10):
        spatial_shape = snapshot_field.shape[1:]  # e.g. (Ny, Nx) or (3, Ny, Nx)
        flat = snapshot_field.reshape(Nt, -1)  # (Nt, prod(spatial_shape))
        b_dist = (W_boot @ flat).reshape((B,) + spatial_shape)  # (B, *spatial_shape)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='All-NaN slice encountered')
            lo = np.nanpercentile(b_dist, 100 * (alpha / 2), axis=0)
            hi = np.nanpercentile(b_dist, 100 * (1 - alpha / 2), axis=0)
        return lo, hi




