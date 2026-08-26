import os
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import warnings

def load_full_npz(f):
    print(f"\r{os.path.basename(f)}", end="")
    d = np.load(f)
    return d["U"], d["V"], d["W"]

def load_fluct_npz(f):
    print(f"\r{os.path.basename(f)}", end="")
    d = np.load(f)
    return d["U_fluct"], d["V_fluct"], d["W_fluct"]

def extract_full(filename):
    files = sorted(glob.glob(filename))

    print('Loading full snapshots... ')
    with ThreadPoolExecutor() as ex:
        results = list(ex.map(load_full_npz, files))

    U, V, W = zip(*results)

    U = np.stack(U)
    V = np.stack(V)
    W = np.stack(W)

    return U, V, W

def extract_fluct(filename):
    files = sorted(glob.glob(filename))

    print('Loading fluctuating snapshots... ')
    with ThreadPoolExecutor() as ex:
        results = list(ex.map(load_fluct_npz, files))

    U_fluct, V_fluct, W_fluct = zip(*results)

    U_fluct = np.stack(U_fluct)
    V_fluct = np.stack(V_fluct)
    W_fluct = np.stack(W_fluct)

    return U_fluct, V_fluct, W_fluct

def bayesian_bootstrap_ci(data, statistic, B=1000, alpha=0.05, rng=None, spatial_chunk=1024):
    """
    Vectorised Bayesian bootstrap CI for any statistic of the form:
        statistic(arrays, g) -> ndarray or scalar
    where g : (B, Nt) is the full batch of Dirichlet weights.

    Parameters
    ----------
    data      : ndarray or tuple of ndarrays, each (Nt, ...)
    statistic : callable  f(arrays, g) -> (B, ...)
    B         : number of replicates
    alpha     : significance level
    """
    if rng is None:
        rng = np.random.default_rng()

    arrays = data if isinstance(data, tuple) else (data,)
    Nt     = arrays[0].shape[0]
    shape  = arrays[0].shape[1:]
    Npts   = int(np.prod(shape))

    u = np.sort(rng.uniform(0, 1, size=(B, Nt - 1)), axis=1)
    g = np.diff(np.concatenate([np.zeros((B, 1)), u, np.ones((B, 1))], axis=1), axis=1)

    flat_arrays = tuple(a.reshape(Nt, -1) for a in arrays)

    # Run one chunk to infer the sample shape (B, ...)
    test_chunk   = tuple(a[:, :1] for a in flat_arrays)
    test_samples = statistic(test_chunk, g)          # e.g. (B, 3, 1)
    extra_dims   = test_samples.shape[1:-1]           # e.g. (3,)

    ci_flat = np.empty((2, *extra_dims, Npts))

    for start in range(0, Npts, spatial_chunk):
        end   = min(start + spatial_chunk, Npts)
        chunk = tuple(a[:, start:end] for a in flat_arrays)
        samples = statistic(chunk, g)                            # (B, [extra_dims,] chunk)
        ci_flat[..., start:end] = np.quantile(samples, [alpha/2, 1 - alpha/2], axis=0)

    ci = ci_flat.reshape(2, *extra_dims, *shape)
    return ci

def single_bayesian_bootstrap_ci(field, statistic, B=1000, alpha=0.05, rng=None):
    """
    Bootstrap CI for statistics that consume a full (Nt, ny, nx) field.
    Returns ci of shape (2, nx).
    """
    if rng is None:
        rng = np.random.default_rng()

    Nt = field.shape[0]
    u  = np.sort(rng.uniform(0, 1, size=(B, Nt - 1)), axis=1)
    g  = np.diff(
        np.concatenate([np.zeros((B, 1)), u, np.ones((B, 1))], axis=1),
        axis=1
    )                                                      # (B, Nt)

    samples = statistic((field,), g)                    # (B, nx)
    return np.quantile(samples, [alpha / 2, 1 - alpha / 2], axis=0)  # (2, nx)

def weighted_mean(arrays, g):
    """
    Weighted ensemble mean for one or more velocity components.

    arrays : tuple of (Nt, Npts_chunk)
    g      : (B, Nt)
    returns: (B, n_components, Npts_chunk)
    """
    return np.stack([g @ arr for arr in arrays], axis=1)   # (B, C, Npts)

def weighted_rms(arrays, g):
    """
    Weighted RMS  sqrt( E[u^2] )  for one or more components.

    arrays : tuple of (Nt, Npts_chunk)
    g      : (B, Nt)
    returns: (B, n_components, Npts_chunk)
    """
    return np.sqrt(np.stack([g @ (arr ** 2) for arr in arrays], axis=1))

def weighted_structure_function(arrays, g):
    """
    Weighted second-order structure function.

    arrays : tuple containing one array of shape (Nt, ny, nx)
    g      : (B, Nt)
    returns: (B, nx)   — D(r) for lags 0 … nx-1
    """
    field = arrays[0]                    # (Nt, ny, nx)
    Nt, ny, nx = field.shape
    B = g.shape[0]

    D = np.zeros((B, nx))
    for r in range(nx):
        u1   = field[:, :, r:]           # (Nt, ny, nx-r)
        u2   = field[:, :, :nx - r]      # (Nt, ny, nx-r)
        diff2 = (u1 - u2) ** 2           # (Nt, ny, nx-r)

        # collapse spatial dims → mean over (ny, nx-r) positions per snapshot
        diff2_mean = diff2.reshape(Nt, -1).mean(axis=1)   # (Nt,)

        # weighted average over snapshots for each bootstrap replicate
        D[:, r] = g @ diff2_mean                           # (B,)

    return D                                               # (B, nx)


case = "6-12_5"

full_folder = "/Volumes/PIV Data1/Stereo_Processed/" + case + "/Full_npz/snap_*.npz"
fluct_folder = "/Volumes/PIV Data1/Stereo_Processed/" + case + "/Fluctuating_npz/snap_*.npz"
avg_file = '/Volumes/PIV Data1/Stereo_Processed/' + case + '/Ensemble_Averages/Averages.npz'



# ── Extract full data ─────────────────────────
U_all, V_all, W_all = extract_full(full_folder)

# ── Extract all fluctuating snapshots ─────────────────────────
U_fluct, V_fluct, W_fluct = extract_fluct(fluct_folder)

# ── Extract averages ─────────────────────────
d = np.load(avg_file)
X, Y, U_mean, V_mean, W_mean, U_rms, V_rms, W_rms, uv, uw, vw = \
    (d["X"], d["Y"], d["U_mean"], d["V_mean"], d["W_mean"], d["U_rms"], d["V_rms"], d["W_rms"], d["uv"], d["uw"], d["vw"])


# --- RMS CI  →  shape (2, 3, ny, nx)
RMS_ci = bayesian_bootstrap_ci((U_fluct, V_fluct, W_fluct), weighted_rms, rng=None)




ci_low_U,  ci_high_U  = RMS_ci[:, 0]   # (Ny, Nx) each
print(np.nanmean(U_rms), np.nanmean(ci_low_U), np.nanmean(ci_high_U))


# 0.08995702344023349