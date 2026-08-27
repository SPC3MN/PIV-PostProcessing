import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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
processed_root = r"/path/to/processed"                       # same root Stereo_Decomposition.py wrote to
results_path = r"/path/to/results/Stereo_Results.xlsx"        # single workbook, regenerated each run
only = None                                                   # limit to one case name, or None for every case found

save = False

INTEGRAL_LENGTH_EXTEND = 10  # multiple of the fitted length scale to extrapolate the model past the data

RESULT_COLUMNS = [
    "case",
    "U_mean", "V_mean", "W_mean",
    "U_rms", "V_rms", "W_rms",
    "Isotropy_ratio_UV", "Isotropy_ratio_UW",
    "TKE_mean", "TKE_perdev",
    "eps_11", "eps_33",
    "L_11", "L_33",
    "lambda_1", "lambda_3",
    "eta2_mean", "xi_mean",
]

case_dirs = discover_case_dirs(
    processed_root, required_glob=os.path.join("Ensemble_Averages", "Averages.npz"))
if only:
    case_dirs = {only: case_dirs[only]}

print(f"Found {len(case_dirs)} case(s).")

rows = []

for case_name, case_dir in case_dirs.items():
    print(f"\n===== {case_name} =====")

    avg_file = os.path.join(case_dir, "Ensemble_Averages", "Averages.npz")
    structure_file = os.path.join(case_dir, "Ensemble_Averages", "Structure_Function.npz")
    auto_file = os.path.join(case_dir, "Ensemble_Averages", "Autocorrelation_Function.npz")
    lumley_file = os.path.join(case_dir, "Lumley_Statistics", "Lumley_Statistics.npz")

    # --------------------------------------------
    #               Load Processed Data
    # --------------------------------------------

    # ── Load average file ───────────

    d = np.load(avg_file)
    X, Y, U_mean, V_mean, W_mean, U_rms, V_rms, W_rms, TKE, uv, uw, vw = (

        d["X"], d["Y"], d["U_mean"], d["V_mean"], d["W_mean"],
        d["U_rms"], d["V_rms"], d["W_rms"], d["TKE"], d["uv"], d["uw"], d["vw"])

    dr = (X[0, 1] - X[0, 0]) / 10

    # ── Load structure function file ───────────
    d = np.load(structure_file)

    conv = 10000  # m^2/s^2 to cm^2/s^2

    D_11 = d["D11"][1:-10] * conv
    D_11_low = d["D11_low"][1:-10] * conv
    D_11_hi = d["D11_hi"][1:-10] * conv

    D_33 = d["D33"][1:-10] * conv
    D_33_low = d["D33_low"][1:-10] * conv
    D_33_hi = d["D33_hi"][1:-10] * conv

    # ── Load autocorrelation function file ───────────
    d = np.load(auto_file)

    rho_11 = d["rho11"][1:-1]
    rho_11_low = d["rho11_low"][1:-1]
    rho_11_hi = d["rho11_hi"][1:-1]

    rho_33 = d["rho33"][1:-1]
    rho_33_low = d["rho33_low"][1:-1]
    rho_33_hi = d["rho33_hi"][1:-1]

    # ── Load lumley file ───────────
    d = np.load(lumley_file)

    eta2 = d['eta2']
    xi = d['xi']
    II = d['II']
    III = d['III']

    plt.contourf(eta2, levels=10)
    plt.show()

    plt.contourf(xi, levels=10)
    plt.show()

    # --------------------------------------------
    #               Result Plotting
    # --------------------------------------------

    # ── TKE contour plot ───────────

    fig, ax = plt.subplots()

    ax.contourf(X, Y, gaussian_filter(TKE, 3), cmap='viridis', levels=10)
    plt.show()

    # ── Structure function plot ───────────

    # ---------------- Find convenient place to put theoretical scaling lines  ----------------
    r1 = np.arange(0, len(D_11)) * dr
    r3 = np.arange(0, len(D_33)) * dr

    # ---------------- Calculate where structure function show correct scaling and plot ----------------

    # Find the derivative of D11 and D33 (smooth first)
    result11 = np.gradient(gaussian_filter(np.log(D_11), 3), np.log(r1))
    result33 = np.gradient(gaussian_filter(np.log(D_33), 3), np.log(r3))

    # create a mask where the slope follows 2/3 scaling +-10%
    mask11 = (result11 > .54) & (result11 < .7)
    mask33 = (result33 > .54) & (result33 < .7)

    max_beg11, max_end11 = Mask_Region(mask11)
    max_beg33, max_end33 = Mask_Region(mask33)

    plt.loglog(r1[1:], D_11[1:], label="D_11", c='r')
    plt.loglog(r3[1:], D_33[1:], label="D_33", c='b')

    plt.xlabel(r'$r$ (cm)')
    plt.ylabel(r'cm^2/s^2')

    plt.legend()
    plt.show()

    lambda_1 = taylor_microscale(r1, rho_11)
    lambda_3 = taylor_microscale(r3, rho_33)

    # ── Dissipation rate plot ───────────

    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)

    C2 = 2

    eps_11 = ((D_11 / C2) ** (3 / 2)) / r1
    eps_11_low = ((D_11_low / C2) ** (3 / 2)) / r1
    eps_11_hi = ((D_11_hi / C2) ** (3 / 2)) / r1

    eps_33 = ((D_33 / C2) ** (3 / 2)) / r3
    eps_33_low = ((D_33_low / C2) ** (3 / 2)) / r3
    eps_33_hi = ((D_33_hi / C2) ** (3 / 2)) / r3

    axes[0].semilogx(r1, eps_11, label='e_11', c='r')
    axes[0].semilogx(r1, eps_11_low, label='CI', c='black', linestyle='--')
    axes[0].semilogx(r1, eps_11_hi, label='CI', c='black', linestyle='--')

    axes[1].semilogx(r3, eps_33, label='e_33', c='b')
    axes[1].semilogx(r3, eps_33_low, label='CI', c='black', linestyle='--')
    axes[1].semilogx(r3, eps_33_hi, label='CI', c='black', linestyle='--')

    axes[0].axhline(np.mean(eps_11[max_beg11:max_end11]), color='r', linestyle='--', label=f'eps = {np.mean(eps_11[max_beg11:max_end11])}')
    axes[1].axhline(np.mean(eps_33[max_beg33:max_end33]), color='b', linestyle='--', label=f'eps = {np.mean(eps_33[max_beg33:max_end33])}')

    axes[0].set_xlabel(r'$r$ (cm)')
    axes[0].set_ylabel(r'$cm^2/s^3$')
    axes[1].set_xlabel(r'$r$ (cm)')
    axes[0].legend()
    axes[1].legend()
    plt.show()

    print(f'eps_11, eps_33 = {np.mean(eps_11[max_beg11:max_end11])}, {np.mean(eps_33[max_beg33:max_end33])}')

    eps_11 = np.mean(eps_11[max_beg11:max_end11])
    eps_33 = np.mean(eps_33[max_beg33:max_end33])

    # ── Integral length plot ───────────

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

    r1 = np.arange(0, len(rho_11)) * dr
    r3 = np.arange(0, len(rho_33)) * dr

    ######  L_11
    fit11 = fit_integral_length(
        rho_11, r1, max_beg11, max_end11, INTEGRAL_LENGTH_EXTEND,
        rho_low=rho_11_low, rho_hi=rho_11_hi)
    L_11 = fit11["L_ext"]

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
    L_data_low = np.trapezoid(fit33["rho_low_trunc"], fit33["r_trunc"])
    L_data_hi = np.trapezoid(fit33["rho_hi_trunc"], fit33["r_trunc"])

    axes[1].plot(fit33["r_trunc"], fit33["rho_trunc"], label=f'Data: L = {fit33["L_data"]} CI= {L_data_low, L_data_hi}')
    axes[1].plot(fit33["r_fit"], fit33["rho_fit"], 'r', label=f'Model fit: L = {fit33["L_fit"]}')
    axes[1].plot(fit33["r_trunc"], fit33["rho_low_trunc"], label=f'CI', linestyle='--')
    axes[1].plot(fit33["r_trunc"], fit33["rho_hi_trunc"], label=f'CI', linestyle='--')
    axes[1].plot(fit33["r_ext"][len(fit33["r_fit"]):], fit33["rho_ext"][len(fit33["r_fit"]):], 'r',
                 label=f'Extended model: L = {fit33["L_ext"]}', linestyle='--')
    axes[1].set_xlabel(r'r (cm)')
    axes[1].legend()
    plt.show()

    # Corner ξ values from boundary intersections
    xi_B = 0.6          # right axisymmetric meets 2-D limit
    xi_C = -0.303        # left axisymmetric meets 2-D limit

    xi_r = np.linspace(0, xi_B, 300)
    eta_r = 6 ** (1/6) * xi_r

    xi_l = np.linspace(xi_C, 0, 300)
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

    # ── Summary row for this case ───────────

    U_mean_avg = round(100 * np.nanmean(U_mean), 3)
    V_mean_avg = round(100 * np.nanmean(V_mean), 3)
    W_mean_avg = round(100 * np.nanmean(W_mean), 3)
    U_rms_avg = round(100 * np.nanmean(U_rms), 3)
    V_rms_avg = round(100 * np.nanmean(V_rms), 3)
    W_rms_avg = round(100 * np.nanmean(W_rms), 3)

    Isotropy_ratio_UV = np.nanmean(U_rms) / np.nanmean(V_rms)
    Isotropy_ratio_UW = np.nanmean(U_rms) / np.nanmean(W_rms)
    TKE_mean = np.nanmean(TKE) * 10000
    TKE_perdev = (np.nanstd(TKE) / np.nanmean(TKE)) * 100

    eta2_mean = np.nanmean(eta2)
    xi_mean = np.nanmean(xi)

    rows.append(dict(zip(RESULT_COLUMNS, [
        case_name,
        U_mean_avg, V_mean_avg, W_mean_avg,
        U_rms_avg, V_rms_avg, W_rms_avg,
        Isotropy_ratio_UV, Isotropy_ratio_UW,
        TKE_mean, TKE_perdev,
        eps_11, eps_33,
        L_11, L_33,
        lambda_1, lambda_3,
        eta2_mean, xi_mean,
    ])))

if save:
    write_results_xlsx(rows, RESULT_COLUMNS, results_path)
