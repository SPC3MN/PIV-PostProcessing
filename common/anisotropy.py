import numpy as np


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
