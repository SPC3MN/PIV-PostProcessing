import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.special import gamma, kv
from matplotlib.patches import Rectangle
import warnings
import pandas as pd

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
def load_npz(f):
    d = np.load(f)
    return d["U_fluct"], d["V_fluct"], d["W_fluct"]
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
def Homogenous_Rect(TKE_field, thresh=0.05):
    TKE_field = gaussian_filter(TKE_field, 1)
    Ny, Nx = TKE_field.shape
    mean = np.nanmean(TKE_field)
    std = np.nanstd(TKE_field)
    n=0

    while std/mean >= thresh:
        n += 1
        TKE = TKE_field[n:Ny-n, 2*n:Nx-2*n]
        mean = np.nanmean(TKE)
        std = np.nanstd(TKE)


    return n
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

# --------------------------------------------
#               Control
# --------------------------------------------
plotting = True
save = False

cases = ['6s-12_5p', '6s-12_5p(t_on=1_5, t_off=0_3)', '6s-12_5p(t_on=1_5, t_off=1_5)']
TKE_mean_list = []
# TKE_max = 0
# TKE_min = 1e6
# TKE_mean_overall =  [np.float64(0.014343049349761045), np.float64(0.012840811017701124), np.float64(0.012591144369736249),
#      np.float64(0.010679178450609306), np.float64(0.011733872248761863), np.float64(0.015765478525245777),
#      np.float64(0.015401407791010801), np.float64(0.014111731891907864), np.float64(0.012518283588228363),
#      np.float64(0.011733872248761863), np.float64(0.015765478525245777), np.float64(0.015491179020728557),
#      np.float64(0.015889295601226126), np.float64(0.015059772731492515)]

for idx, case in enumerate(cases):

    avg_file = '/Users/alecsangster/Desktop/Swirl_Planar_Processed/' + case + '/Ensemble_Averages/Averages.npz'
    structure_file = '/Users/alecsangster/Desktop/Swirl_Planar_Processed/' + case + '/Ensemble_Averages/Structure_Function.npz'
    auto_file = '/Users/alecsangster/Desktop/Swirl_Planar_Processed/' + case + '/Ensemble_Averages/Autocorrelation_Function.npz'
    energy_file = '/Users/alecsangster/Desktop/Swirl_Planar_Processed/' + case + '/Ensemble_Averages/Energy_Spectra.npz'

    print(case)

    # d = np.load(energy_file)
    # kx, ky, Eu, Ev = d['kx'], d['ky'], d['Eu'], d['Ev']
    # #
    # # # plt.figure(figsize=(6, 5))
    # # # plt.loglog(k, Ev, lw=2)
    # # # plt.xlabel(r'Wavenumber, $k$ [rad/m]')
    # # # plt.ylabel(r'Energy Spectrum, $E_{11}(k)$')
    # # # plt.grid(True, which='both', alpha=0.3)
    # k_res = np.pi/(70/1000)
    # k_ref = kx  # choose a range in the inertial subrange
    # C = Eu[40] * kx[40] ** (5 / 3)  # scale the line to your data
    # plt.loglog(kx, Eu)
    # plt.loglog(ky, Ev)
    # plt.loglog(k_ref, C * k_ref ** (-5 / 3), 'k--', label=r'$k^{-5/3}$')
    # plt.axvline(
    #     k_res,
    #     linestyle='--',
    #     label=fr'$k_{{res}}={k_res:.0f}\ \mathrm{{m^{{-1}}}}$'
    # )
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # --------------------------------------------
    #               Load Processed Data
    # --------------------------------------------

    # ── Load average file ───────────

    d = np.load(avg_file)
    X, Y, U_mean, V_mean, U_rms, V_rms, TKE = (

        d["X"], d["Y"], d["U_mean"], d["V_mean"],
        d["U_rms"], d["V_rms"], d["TKE"] )
    print((abs(X[0][0] - X[0][1])))
    # if np.nanmax(TKE) > TKE_max:
    #     TKE_max = np.nanmax(TKE)
    #
    # if np.nanmin(TKE) < TKE_min:
    #     TKE_min = np.nanmin(TKE)

    dr = (X[0, 1] - X[0, 0])/10 # mm to cm

    U_mean_avg = round(100*np.nanmean(U_mean), 3)
    V_mean_avg = round(100*np.nanmean(V_mean), 3)
    U_rms_avg = round(100*np.nanmean(U_rms), 3)
    V_rms_avg = round(100*np.nanmean(V_rms), 3)


    # ── Load structure function file ───────────
    d = np.load(structure_file)

    conv = 10000 #m^2/s^2 to cm^2/s^2

    D_11 = d["D11"]*conv
    D_11_low = d["D11_low"]*conv
    D_11_hi = d["D11_hi"]*conv

    D_33 = d["D33"]*conv
    D_33_low = d["D33_low"]*conv
    D_33_hi = d["D33_hi"]*conv

    # ── Load autocorrelation function file ───────────
    d = np.load(auto_file)

    rho_11 = d["rho11"]
    rho_11_low = d["rho11_low"]
    rho_11_hi = d["rho11_hi"]

    rho_33 = d["rho33"]
    rho_33_low = d["rho33_low"]
    rho_33_hi = d["rho33_hi"]

    Isotropy_ratio = np.nanmean(U_rms)/np.nanmean(V_rms)
    M1 = (2*np.nanmean(abs(U_mean))+np.nanmean(abs(V_mean))) / (2*np.nanmean(abs(U_rms))+np.nanmean(abs(V_rms)))
    M2 = (2*np.nanmean(U_mean)**2+np.nanmean(V_mean)**2) / (2*np.nanmean(U_rms)**2+np.nanmean(V_rms)**2)
    TKE_mean =  np.nanmean(TKE)*10000
    TKE_mean_list.append(np.nanmean(TKE))
    print(f'TKE_mean= {TKE_mean}')


    TKE_perdev = (np.nanstd(TKE)/np.nanmean(TKE))*100

    # --------------------------------------------
    #               Result Plotting
    # --------------------------------------------


    # ── TKE contour plot ───────────
    if plotting:
        fig, ax = plt.subplots()

        ax.contourf(X, Y, gaussian_filter(TKE, 3), cmap='viridis', levels=10)


        # n = Homogenous_Rect(gaussian_filter(TKE, 3))
        # X_trim = X[n:-n, 2*n:-2*n]
        # Y_trim = Y[n:-n, 2*n:-2*n]
        #
        # x0 = X_trim[0, 0]
        # y0 = Y_trim[0, 0]
        #
        # width  = X_trim[0, -1] - X_trim[0, 0]
        # height = Y_trim[-1, 0] - Y_trim[0, 0]
        #
        # rect = Rectangle(
        #     (x0, y0),
        #     width,
        #     height,
        #     edgecolor='red',
        #     facecolor='none',
        #     linewidth=2
        # )
        #
        # ax.add_patch(rect)
        plt.show()


    # ── Structure function plot ───────────

    # ---------------- Find convenient place to put theoretical scaling lines  ----------------
    r1 = np.arange(0, len(D_11))*dr
    r3 = np.arange(0, len(D_33))*dr


    # ---------------- Calculate where structure function show correct scaling and plot ----------------
    # Find the derivative of D11 and D33 (smooth first)
    result11 = np.gradient(np.log(D_11[1:]), np.log(r1[1:]))
    result33 = np.gradient(np.log(D_33[1:]), np.log(r3[1:]))


    # create a mask where the slope follows 2/3 scaling +-10%
    mask11 = (result11 > 0.6) & (result11 < 0.73)
    mask33 = (result33 > 0.6) & (result33 < 0.73)

    max_beg11, max_end11 = Mask_Region(mask11)
    max_beg33, max_end33 = Mask_Region(mask33)

    if plotting:
        plt.axvspan(r1[max_beg11], r1[max_end11], alpha=0.2, color='r', label='inertial range (D_11)')
        plt.axvspan(r3[max_beg33], r3[max_end33], alpha=0.2, color='b', label='inertial range (D_33)')

        # # create a mask where the slope follows 2/3 scaling +-10%
        # mask112 = (result11 > 1.8) & (result11 < 2.2)
        # mask332 = (result33 > 1.8) & (result33 < 2.2)
        #
        # max_beg112, max_end112 = Mask_Region(mask112)
        # max_beg332, max_end332 = Mask_Region(mask332)
        #
        # # plt.axvspan(r1[max_beg112], r1[max_end112], alpha=0.2, color='r', label='dissipation range (D_11)')
        # plt.axvspan(r3[max_beg332], r3[max_end332], alpha=0.2, color='b', label='dissipation range (D_33)')

        plt.loglog(r1[1:], D_11[1:], label="D_11", c='r')
        plt.loglog(r1[1:], D_11_low[1:], label="CI", c='r', linestyle='--')
        plt.loglog(r1[1:], D_11_hi[1:], label="CI", c='r', linestyle='--')
        #
        # plt.loglog(r3[1:], D_33[1:], label="D_33", c='b')
        # plt.loglog(r3[1:], D_33_low[1:], label="CI", c='b', linestyle='--')
        # plt.loglog(r3[1:], D_33_hi[1:], label="CI", c='b', linestyle='--')
        # plt.loglog(r3[1:], D_33[1:], label="D_33", c='b')



        plt.xlabel(r'$r$ (cm)')
        plt.ylabel(r'cm^2/s^2')

        plt.legend()
        plt.show()

    lambda_1 = taylor_microscale(r1, rho_11)
    lambda_3 = taylor_microscale(r3, rho_33)


    # ── Dissipation rate plot ───────────


    C2 = 2

    eps_11 = ( (D_11/C2)**(3/2) ) / r1
    eps_11_low = ( (D_11_low/C2)**(3/2) ) / r1
    eps_11_hi = ( (D_11_hi/C2)**(3/2) ) / r1

    eps_33 = ( (D_33/C2)**(3/2) ) / r3
    eps_33_low = ( (D_33_low/C2)**(3/2) ) / r3
    eps_33_hi = ( (D_33_hi/C2)**(3/2) ) / r3

    if plotting:
        fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)

        axes[0].semilogx(r1, eps_11, label='e_11', c='r')
        axes[0].semilogx(r1, eps_11_low, label='CI', c='black', linestyle='--')
        axes[0].semilogx(r1, eps_11_hi, label='CI', c='black', linestyle='--')

        axes[1].semilogx(r3, eps_33, label='e_33', c='b')
        axes[1].semilogx(r3, eps_33_low, label='CI', c='black', linestyle='--')
        axes[1].semilogx(r3, eps_33_hi, label='CI', c='black', linestyle='--')

        axes[0].axvspan(r1[max_beg11], r1[max_end11], alpha=0.2, color='r', label='inertial range (D_11)')
        axes[1].axvspan(r3[max_beg33], r1[max_end33], alpha=0.2, color='b', label='inertial range (D_33)')

        axes[0].axhline(np.mean(eps_11[max_beg11:max_end11]), color='r', linestyle='--', label=f'eps = {np.mean(eps_11[max_beg11:max_end11])}')
        axes[1].axhline(np.mean(eps_33[max_beg33:max_end33]), color='b', linestyle='--', label=f'eps = {np.mean(eps_33[max_beg33:max_end33])}')

        axes[0].set_xlabel(r'$r$ (cm)')
        axes[0].set_ylabel(r'$cm^2/s^3$')
        axes[1].set_xlabel(r'$r$ (cm)')
        axes[0].legend()
        axes[1].legend()
        plt.show()

    eps_11 = np.mean(eps_11[max_beg11:max_end11])
    eps_33 = np.mean(eps_33[max_beg33:max_end33])


    # ── Integral length plot ───────────



    # Model function
    def model(r, q, Lm):
        alpha = np.sqrt(np.pi) * gamma(q + 0.5) / gamma(q)

        x = (r / Lm) * alpha

        prefactor = 2 / gamma(q)
        term = ((r * alpha) / (2 * Lm)) ** q

        return prefactor * term * kv(q, x)


    r1 = np.arange(0, len(rho_11))*dr
    r3 = np.arange(0, len(rho_33))*dr


    ######  L_11
    rho_ij, rho_ij_low, rho_ij_hi, rj = rho_11, rho_11_low, rho_11_hi, r1


    max_begij, max_endij = max_beg11, max_end11

    rho = rho_ij[max_begij:max_endij]
    r = rj[max_begij:max_endij]

    p0 = [1.0, np.mean(r)]

    params, cov = params, cov = curve_fit(model, r, rho, p0=[1.0, np.mean(r)], bounds=([0.1, 0], [10, np.inf])
    )

    q_fit, Lm_fit = params
    r_fit = np.linspace(min(rj[1:]), max(rj), 500)
    rho_fit = model(r_fit, q_fit, Lm_fit)

    L_data = np.trapezoid(rho_ij, rj)
    L_fit = np.trapezoid(rho_fit, r_fit)

    r_ext = np.linspace(min(rj[1:]), 5*Lm_fit, 1500)  # extend a few length scales
    rho_ext = model(r_ext, q_fit, Lm_fit)
    L_ext = np.trapezoid(rho_ext, r_ext)

    first_zero = np.argmax(rho_ij < 0)
    if first_zero > 0:
        rho_ij = rho_ij[:first_zero]
        rho_ij_low = rho_ij_low[:first_zero]
        rho_ij_hi = rho_ij_hi[:first_zero]
        rj = rj[:first_zero]
    L_data = np.trapezoid(rho_ij, rj)

    L_11 = L_ext

    if plotting:
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

        axes[0].plot(rj, rho_ij, label=f'Data: L = {L_data}')
        axes[0].plot(rj, rho_ij_low, label=f'CI', linestyle='--')
        axes[0].plot(rj, rho_ij_hi, label=f'CI', linestyle='--')
        axes[0].plot(r_fit, rho_fit, 'r', label=f'Model fit: L = {L_fit}')
        axes[0].plot(r_ext[len(r_fit):], rho_ext[len(r_fit):], 'r', label=f'Extended model: L = {L_ext}', linestyle='--')
        axes[0].set_ylabel(r'Autocorrelation Function $\rho_{ij}(r)$')
        axes[0].set_xlabel(r'r (mm)')
        axes[0].legend()

    ######  L_33
    rho_ij, rho_ij_low, rho_ij_hi, rj = rho_33, rho_33_low, rho_33_hi, r3

    max_begij, max_endij = max_beg11, max_end11

    rho = rho_ij[max_begij:max_endij]
    r = rj[max_begij:max_endij]

    p0 = [1.0, np.mean(r)]

    params, cov = params, cov = curve_fit(model, r, rho, p0=[1.0, np.mean(r)], bounds=([0.1, 0], [10, np.inf])
    )

    q_fit, Lm_fit = params

    r_fit = np.linspace(min(rj[1:]), max(rj), 500)
    rho_fit = model(r_fit, q_fit, Lm_fit)


    L_fit = np.trapezoid(rho_fit, r_fit)

    r_ext = np.linspace(min(rj[1:]), 5*Lm_fit, 1500)  # extend a few length scales
    rho_ext = model(r_ext, q_fit, Lm_fit)
    L_ext = np.trapezoid(rho_ext, r_ext)

    idx = np.argmax(rj)


    first_zero = np.argmax(rho_ij < 0)
    if first_zero > 0:
        rho_ij = rho_ij[:first_zero]
        rho_ij_low = rho_ij_low[:first_zero]
        rho_ij_hi = rho_ij_hi[:first_zero]
        rj = rj[:first_zero]

    L_data = np.trapezoid(rho_ij, rj)

    L_33 = L_ext

    if plotting:

        axes[1].plot(rj, rho_ij, label=f'Data: L = {L_data}')
        axes[1].plot(r_fit, rho_fit, 'r', label=f'Model fit: L = {L_fit}')
        axes[1].plot(rj, rho_ij_low, label=f'CI', linestyle='--')
        axes[1].plot(rj, rho_ij_hi, label=f'CI', linestyle='--')
        axes[1].plot(r_ext[len(r_fit):], rho_ext[len(r_fit):], 'r', label=f'Extended model: L = {L_ext}', linestyle='--')
        axes[1].set_xlabel(r'r (cm)')
        axes[1].legend()
        plt.show()


    if save:
        Results = [case, U_mean_avg, V_mean_avg, U_rms_avg, V_rms_avg,
         Isotropy_ratio, M1, TKE_mean, TKE_perdev,
         eps_11, eps_33, L_11, L_33, lambda_1, lambda_3]

        existing = pd.read_excel("/Users/alecsangster/Desktop/Final_Planar_Results/Planar_Results(10cm).xlsx")
        new_row = pd.DataFrame([Results], columns=existing.columns)
        updated = pd.concat([existing, new_row], ignore_index=True)
        updated.to_excel("/Users/alecsangster/Desktop/Final_Planar_Results/Planar_Results(10cm).xlsx", index=False)
