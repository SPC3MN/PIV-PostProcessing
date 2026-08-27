import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.special import gamma, kv


def taylor_microscale(lag, corr, points=2):
    lag = np.asarray(lag)
    corr = np.asarray(corr)

    # r window of size points
    r_fit = lag[1:points+1]
    f_fit = corr[1:points+1]

    # quadratic fit
    c2, c1, c0 = np.polyfit(r_fit, f_fit, 2)

    if c2 >= 0:
        raise ValueError("Parabola is not concave down; adjust fit window.")

    return np.sqrt(-1 / c2)


def Mask_Region(mask):
    max_len = 0
    max_start = None
    max_end = None

    current_start = None
    for i, val in enumerate(mask):
        if val:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                length = i - current_start
                if length > max_len:
                    max_len = length
                    max_start = current_start
                    max_end = i
                current_start = None

    # handle case where array ends in True
    if current_start is not None:
        length = len(mask) - current_start
        if length > max_len:
            max_len = length
            max_start = current_start
            max_end = len(mask)

    return max_start, max_end


def Homogenous_Rect(TKE_field, thresh=0.05, sigma=5, verbose=False):
    TKE_field = gaussian_filter(TKE_field, sigma)
    Ny, Nx = TKE_field.shape
    mean = np.nanmean(TKE_field)
    std = np.nanstd(TKE_field)
    n = 0

    while std/mean >= thresh:
        if verbose:
            print(std / mean)
        n += 1
        TKE = TKE_field[n:Ny-n, 2*n:Nx-2*n]
        mean = np.nanmean(TKE)
        std = np.nanstd(TKE)

    return n


def autocorrelation_model(r, q, Lm):
    """Lumley-Newman-style model for a normalized two-point autocorrelation,
    fit by `fit_integral_length` below."""
    alpha = np.sqrt(np.pi) * gamma(q + 0.5) / gamma(q)

    x = (r / Lm) * alpha

    prefactor = 2 / gamma(q)
    term = ((r * alpha) / (2 * Lm)) ** q

    return prefactor * term * kv(q, x)


def fit_integral_length(rho, r, mask_beg, mask_end, extend_factor, rho_low=None, rho_hi=None):
    """Fit `autocorrelation_model` to rho(r) over the inertial-range window
    [mask_beg:mask_end), extrapolate `extend_factor` fitted length scales past
    the data, and integrate (trapezoid) for the integral length scale.

    `extend_factor` is intentionally a required argument, not a shared
    default: the two analysis scripts historically extended the model by a
    different multiple of the fitted length scale (5x vs 10x), and this
    preserves each script's own numbers rather than silently unifying them.

    Returns a dict with the fit/extension curves and L_data/L_fit/L_ext, plus
    (if rho_low/rho_hi were passed) the same first-negative-lag truncation
    applied to them that `rho`/`r` receive, for plotting confidence bands.
    """
    rho = np.asarray(rho)
    r = np.asarray(r)

    rho_window = rho[mask_beg:mask_end]
    r_window = r[mask_beg:mask_end]

    p0 = [1.0, np.mean(r_window)]
    params, cov = curve_fit(
        autocorrelation_model, r_window, rho_window, p0=p0, bounds=([0.1, 0], [10, np.inf]))
    q_fit, Lm_fit = params

    r_fit = np.linspace(min(r[1:]), max(r), 500)
    rho_fit = autocorrelation_model(r_fit, q_fit, Lm_fit)
    L_fit = np.trapezoid(rho_fit, r_fit)

    r_ext = np.linspace(min(r[1:]), extend_factor * Lm_fit, 1500)
    rho_ext = autocorrelation_model(r_ext, q_fit, Lm_fit)
    L_ext = np.trapezoid(rho_ext, r_ext)

    first_zero = np.argmax(rho < 0)
    if first_zero > 0:
        r = r[:first_zero]
        if rho_low is not None:
            rho_low = rho_low[:first_zero]
        if rho_hi is not None:
            rho_hi = rho_hi[:first_zero]
        rho = rho[:first_zero]
    L_data = np.trapezoid(rho, r)

    return {
        "q_fit": q_fit, "Lm_fit": Lm_fit,
        "r_fit": r_fit, "rho_fit": rho_fit,
        "r_ext": r_ext, "rho_ext": rho_ext,
        "L_data": L_data, "L_fit": L_fit, "L_ext": L_ext,
        "rho_trunc": rho, "r_trunc": r,
        "rho_low_trunc": rho_low, "rho_hi_trunc": rho_hi,
    }
