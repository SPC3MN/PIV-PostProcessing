import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import glob


def Center_Contour(X, Y, U, V, W, width=300, height=150):
    x_center = X[0, :].mean()
    y_center = Y[:, 0].mean()

    xmask = np.abs(X[0, :] - x_center) <= width / 2
    ymask = np.abs(Y[:, 0] - y_center) <= height / 2

    idx = np.ix_(ymask, xmask)

    X_c = X[idx] - x_center
    Y_c = Y[idx] - y_center

    return X_c, Y_c, U[idx], V[idx], W[idx]


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

def load_npz(f):
    d = np.load(f)
    return d["U_fluct"], d["V_fluct"], d["W_fluct"]


# print('Loading averages... ')
# # ── Extract averages ─────────────────────────
# file = "/Volumes/PIV Data1/Stereo1(6-12_5)1-1/mean_npz/Averages.npz"
#
# d = np.load(file)
# X, Y, U_mean, V_mean, W_mean, U_rms, V_rms, W_rms, uv, uw, vw = (
#
#     d["X"], d["Y"], d["U_mean"], d["V_mean"], d["W_mean"], d["U_rms"], d["V_rms"], d["W_rms"], d["uv"], d["uw"], d["vw"])
#
#
# _, _, U_rms, V_rms, W_rms = Center_Contour(X, Y, U_rms, V_rms, W_rms, width=150)
# X, Y, uv, uw, vw = Center_Contour(X, Y, uv, uw, vw, width=150)



# # ── Extract all fluctuating snapshots ─────────────────────────
# files = sorted(glob.glob("/Volumes/PIV Data1/Initial_Stereo/3s-Burst-Test/fluct_npz/snap_*.npz"))
#
# print('Loading fluctuating snapshots... ')
# with ThreadPoolExecutor() as ex:
#     results = list(ex.map(load_npz, files))
#
# U_fluct, V_fluct, W_fluct = zip(*results)
#
# U_fluct = np.stack(U_fluct)
# V_fluct = np.stack(V_fluct)
# W_fluct = np.stack(W_fluct)

# print('Generating reynolds stress tensor... ')
# R, k = reynolds_stress(U_rms, V_rms, W_rms, uv, uw, vw)
#
# print('Calculating anisotropy invariants... ')
# eta2, xi, II, III = compute_anisotropy_invariants(R, k)
#
# print(f'uu, vv, ww = {np.nanmean(U_rms ** 2), np.nanmean(V_rms ** 2), np.nanmean(W_rms ** 2)}')
# print(f'eta_avg = {np.nanmean(eta2)}')

# Corner ξ values from boundary intersections
xi_B =  0.6          # right axisymmetric meets 2-D limit
xi_C = -0.303       # left axisymmetric meets 2-D limit

xi_r = np.linspace(0, xi_B, 300)
eta_r = 6 ** (1/6) * xi_r

xi_l = np.linspace(xi_C, 0,  300)
eta_l = -6 ** (1/6) * xi_l

xi_t = np.linspace(xi_C, xi_B, 300)
eta_t = np.sqrt(2/9 + 2 * xi_t**3)


plt.plot(xi_r, eta_r, linestyle='--', c='grey')
plt.plot(xi_l, eta_l, linestyle='--', c='grey')
plt.plot(xi_t, eta_t, linestyle='--', c='grey')

plt.scatter(xi.ravel(), np.sqrt(eta2.ravel()), s=1, alpha=0.05)

plt.axvline(0, color='gray', lw=0.5, ls='--')
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\eta$')
plt.title('Lumley Triangle')
plt.tight_layout()
plt.show()