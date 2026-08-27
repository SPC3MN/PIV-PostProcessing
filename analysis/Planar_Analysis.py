import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.discovery import discover_case_dirs
from common.analysis_stats import Mask_Region, taylor_microscale, fit_integral_length
from common.results_io import write_results_xlsx

warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

# --------------------------------------------
#               Control
# --------------------------------------------
processed_root = r"/path/to/processed"                      # same root Planar_Decomposition.py wrote to
results_path = r"/path/to/results/Planar_Results.xlsx"       # single workbook, regenerated each run
only = None                                                  # limit to one case name, or None for every case found

plotting = True
save = False

INTEGRAL_LENGTH_EXTEND = 5  # multiple of the fitted length scale to extrapolate the model past the data

RESULT_COLUMNS = [
    "case", "U_mean", "V_mean", "U_rms", "V_rms",
    "Isotropy_ratio", "M1", "TKE_mean", "TKE_perdev",
    "eps_11", "eps_33", "L_11", "L_33", "lambda_1", "lambda_3",
]

case_dirs = discover_case_dirs(
    processed_root, required_glob=os.path.join("Ensemble_Averages", "Averages.npz"))
if only:
    case_dirs = {only: case_dirs[only]}

print(f"Found {len(case_dirs)} case(s).")

TKE_mean_list = []
rows = []

for case_name, case_dir in case_dirs.items():

    avg_file = os.path.join(case_dir, "Ensemble_Averages", "Averages.npz")
    structure_file = os.path.join(case_dir, "Ensemble_Averages", "Structure_Function.npz")
    auto_file = os.path.join(case_dir, "Ensemble_Averages", "Autocorrelation_Function.npz")
    energy_file = os.path.join(case_dir, "Ensemble_Averages", "Energy_Spectra.npz")

    print(case_name)

    # d = np.load(energy_file)
    # kx, ky, Eu, Ev = d['kx'], d['ky'], d['Eu'], d['Ev']
    # k_res = np.pi/(70/1000)
    # k_ref = kx  # choose a range in the inertial subrange
    # C = Eu[40] * kx[40] ** (5 / 3)  # scale the line to your data
    # plt.loglog(kx, Eu)
    # plt.loglog(ky, Ev)
    # plt.loglog(k_ref, C * k_ref ** (-5 / 3), 'k--', label=r'$k^{-5/3}$')
    # plt.axvline(k_res, linestyle='--', label=fr'$k_{{res}}={k_res:.0f}\ \mathrm{{m^{{-1}}}}$')
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
        d["U_rms"], d["V_rms"], d["TKE"])
    print((abs(X[0][0] - X[0][1])))

    dr = (X[0, 1] - X[0, 0]) / 10  # mm to cm

    U_mean_avg = round(100 * np.nanmean(U_mean), 3)
    V_mean_avg = round(100 * np.nanmean(V_mean), 3)
    U_rms_avg = round(100 * np.nanmean(U_rms), 3)
    V_rms_avg = round(100 * np.nanmean(V_rms), 3)

    # ── Load structure function file ───────────
    d = np.load(structure_file)

    conv = 10000  # m^2/s^2 to cm^2/s^2

    D_11 = d["D11"] * conv
    D_11_low = d["D11_low"] * conv
    D_11_hi = d["D11_hi"] * conv

    D_33 = d["D33"] * conv
    D_33_low = d["D33_low"] * conv
    D_33_hi = d["D33_hi"] * conv

    # ── Load autocorrelation function file ───────────
    d = np.load(auto_file)

    rho_11 = d["rho11"]
    rho_11_low = d["rho11_low"]
    rho_11_hi = d["rho11_hi"]

    rho_33 = d["rho33"]
    rho_33_low = d["rho33_low"]
    rho_33_hi = d["rho33_hi"]

    Isotropy_ratio = np.nanmean(U_rms) / np.nanmean(V_rms)
    M1 = (2 * np.nanmean(abs(U_mean)) + np.nanmean(abs(V_mean))) / (2 * np.nanmean(abs(U_rms)) + np.nanmean(abs(V_rms)))
    M2 = (2 * np.nanmean(U_mean) ** 2 + np.nanmean(V_mean) ** 2) / (2 * np.nanmean(U_rms) ** 2 + np.nanmean(V_rms) ** 2)
    TKE_mean = np.nanmean(TKE) * 10000
    TKE_mean_list.append(np.nanmean(TKE))
    print(f'TKE_mean= {TKE_mean}')

    TKE_perdev = (np.nanstd(TKE) / np.nanmean(TKE)) * 100

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
    r1 = np.arange(0, len(D_11)) * dr
    r3 = np.arange(0, len(D_33)) * dr

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

        plt.loglog(r1[1:], D_11[1:], label="D_11", c='r')
        plt.loglog(r1[1:], D_11_low[1:], label="CI", c='r', linestyle='--')
        plt.loglog(r1[1:], D_11_hi[1:], label="CI", c='r', linestyle='--')

        plt.xlabel(r'$r$ (cm)')
        plt.ylabel(r'cm^2/s^2')

        plt.legend()
        plt.show()

    lambda_1 = taylor_microscale(r1, rho_11)
    lambda_3 = taylor_microscale(r3, rho_33)

    # ── Dissipation rate plot ───────────

    C2 = 2

    eps_11 = ((D_11 / C2) ** (3 / 2)) / r1
    eps_11_low = ((D_11_low / C2) ** (3 / 2)) / r1
    eps_11_hi = ((D_11_hi / C2) ** (3 / 2)) / r1

    eps_33 = ((D_33 / C2) ** (3 / 2)) / r3
    eps_33_low = ((D_33_low / C2) ** (3 / 2)) / r3
    eps_33_hi = ((D_33_hi / C2) ** (3 / 2)) / r3

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

    r1 = np.arange(0, len(rho_11)) * dr
    r3 = np.arange(0, len(rho_33)) * dr

    ######  L_11
    fit11 = fit_integral_length(
        rho_11, r1, max_beg11, max_end11, INTEGRAL_LENGTH_EXTEND,
        rho_low=rho_11_low, rho_hi=rho_11_hi)
    L_11 = fit11["L_ext"]

    if plotting:
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

        axes[0].plot(fit11["r_trunc"], fit11["rho_trunc"], label=f'Data: L = {fit11["L_data"]}')
        axes[0].plot(fit11["r_trunc"], fit11["rho_low_trunc"], label=f'CI', linestyle='--')
        axes[0].plot(fit11["r_trunc"], fit11["rho_hi_trunc"], label=f'CI', linestyle='--')
        axes[0].plot(fit11["r_fit"], fit11["rho_fit"], 'r', label=f'Model fit: L = {fit11["L_fit"]}')
        axes[0].plot(fit11["r_ext"][len(fit11["r_fit"]):], fit11["rho_ext"][len(fit11["r_fit"]):], 'r',
                     label=f'Extended model: L = {fit11["L_ext"]}', linestyle='--')
        axes[0].set_ylabel(r'Autocorrelation Function $\rho_{ij}(r)$')
        axes[0].set_xlabel(r'r (mm)')
        axes[0].legend()

    ######  L_33
    fit33 = fit_integral_length(
        rho_33, r3, max_beg11, max_end11, INTEGRAL_LENGTH_EXTEND,
        rho_low=rho_33_low, rho_hi=rho_33_hi)
    L_33 = fit33["L_ext"]

    if plotting:
        axes[1].plot(fit33["r_trunc"], fit33["rho_trunc"], label=f'Data: L = {fit33["L_data"]}')
        axes[1].plot(fit33["r_fit"], fit33["rho_fit"], 'r', label=f'Model fit: L = {fit33["L_fit"]}')
        axes[1].plot(fit33["r_trunc"], fit33["rho_low_trunc"], label=f'CI', linestyle='--')
        axes[1].plot(fit33["r_trunc"], fit33["rho_hi_trunc"], label=f'CI', linestyle='--')
        axes[1].plot(fit33["r_ext"][len(fit33["r_fit"]):], fit33["rho_ext"][len(fit33["r_fit"]):], 'r',
                     label=f'Extended model: L = {fit33["L_ext"]}', linestyle='--')
        axes[1].set_xlabel(r'r (cm)')
        axes[1].legend()
        plt.show()

    rows.append(dict(zip(RESULT_COLUMNS, [
        case_name, U_mean_avg, V_mean_avg, U_rms_avg, V_rms_avg,
        Isotropy_ratio, M1, TKE_mean, TKE_perdev,
        eps_11, eps_33, L_11, L_33, lambda_1, lambda_3,
    ])))

if save:
    write_results_xlsx(rows, RESULT_COLUMNS, results_path)
