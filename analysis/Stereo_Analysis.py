import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.special import gamma, kv
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")


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

def Center_2dArray(X, Y, U, width=300, height=150):
    x_center = X[0, :].mean()
    y_center = Y[:, 0].mean()

    xmask = np.abs(X[0, :] - x_center) <= width / 2
    ymask = np.abs(Y[:, 0] - y_center) <= height / 2

    idx = np.ix_(ymask, xmask)

    X_c = X[idx] - x_center
    Y_c = Y[idx] - y_center

    return X_c, Y_c, U[idx]

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

def load_npz(f):
    d = np.load(f)
    return d["U_fluct"], d["V_fluct"], d["W_fluct"]

def Homogenous_Rect(TKE_field, thresh=0.05):
    TKE_field = gaussian_filter(TKE_field, 5)
    Ny, Nx = TKE_field.shape
    mean = np.nanmean(TKE_field)
    std = np.nanstd(TKE_field)
    n=0

    while std/mean >= thresh:
        print(std / mean)
        n += 1
        TKE = TKE_field[n:Ny-n, 2*n:Nx-2*n]
        mean = np.nanmean(TKE)
        std = np.nanstd(TKE)


    return n


# --------------------------------------------
#               Control
# --------------------------------------------
case = '6-12_5-1-1'
avg_file = "/Volumes/PIV Data1/Stereo_Processed/" + case + "/Ensemble_Averages/Averages.npz"
structure_file = "/Volumes/PIV Data1/Stereo_Processed/" + case + "/Ensemble_Averages/Structure_Function.npz"
auto_file = "/Volumes/PIV Data1/Stereo_Processed/" + case + "/Ensemble_Averages/Autocorrelation_Function.npz"
lumley_file = "/Volumes/PIV Data1/Stereo_Processed/" + case + "/Lumley_Statistics/Lumley_Statistics.npz"
boot_file = '/Volumes/PIV Data1/Stereo_Processed/' + case + '/Bootstrap/Bootstrapped_Statistics.npz'


# --------------------------------------------
#               Load Processed Data
# --------------------------------------------

# ── Load average file ───────────

d = np.load(avg_file)
X, Y, U_mean, V_mean, W_mean, U_rms, V_rms, W_rms, TKE, uv, uw, vw = (

    d["X"], d["Y"], d["U_mean"], d["V_mean"], d["W_mean"],
    d["U_rms"], d["V_rms"], d["W_rms"], d["TKE"], d["uv"], d["uw"], d["vw"])

dr = (X[0, 1] - X[0, 0])/10

# ── Load structure function file ───────────
d = np.load(structure_file)

conv = 10000 #m^2/s^2 to cm^2/s^2

D_11 = d["D11"][1:-10]*conv
D_11_low = d["D11_low"][1:-10]*conv
D_11_hi = d["D11_hi"][1:-10]*conv

D_33 = d["D33"][1:-10]*conv
D_33_low = d["D33_low"][1:-10]*conv
D_33_hi = d["D33_hi"][1:-10]*conv

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

X, Y, TKE = Center_2dArray(X, Y, TKE)

ax.contourf(X, Y, gaussian_filter(TKE, 3), cmap='viridis', levels=10)

n = Homogenous_Rect(gaussian_filter(TKE, 5))
X_trim = X[n:-n, 2*n:-2*n]
Y_trim = Y[n:-n, 2*n:-2*n]

x0 = X_trim[0, 0]
y0 = Y_trim[0, 0]

width  = X_trim[0, -1] - X_trim[0, 0]
height = Y_trim[-1, 0] - Y_trim[0, 0]

rect = Rectangle(
    (x0, y0),
    width,
    height,
    edgecolor='red',
    facecolor='none',
    linewidth=2
)

ax.add_patch(rect)
plt.show()


# ── Structure function plot ───────────

# ---------------- Find convenient place to put theoretical scaling lines  ----------------
r1 = np.arange(0, len(D_11))*dr
r3 = np.arange(0, len(D_33))*dr


# ---------------- Calculate where structure function show correct scaling and plot ----------------

# Find the derivative of D11 and D33 (smooth first)
result11 = np.gradient(gaussian_filter(np.log(D_11), 3), np.log(r1))
result33 = np.gradient(gaussian_filter(np.log(D_33), 3), np.log(r3))

# create a mask where the slope follows 2/3 scaling +-10%
mask11 = (result11 > .54) & (result11 < .7)
mask33 = (result33 > .54) & (result33 < .7)

max_beg11, max_end11 = Mask_Region(mask11)
max_beg33, max_end33 = Mask_Region(mask33)


# plt.axvspan(r1[max_beg11], r1[max_end11], alpha=0.2, color='r', label='inertial range (D_11)')
# plt.axvspan(r3[max_beg33], r1[max_end33], alpha=0.2, color='b', label='inertial range (D_33)')

plt.loglog(r1[1:], D_11[1:], label="D_11", c='r')
plt.loglog(r3[1:], D_33[1:], label="D_33", c='b')

# plt.loglog(
#     r1[mid_idx:],
#     3*C * r1[mid_idx:]**(2/3), linestyle='--',
#     color='black',
#     label=r"$r^{2/3}$"
# )
#
# plt.loglog(
#     r1[:mid_idx],
#     0.5*C * r1[:mid_idx]**2, linestyle='--',
#     color='gray',
#     label=r"$r^{2}$"
# )


plt.xlabel(r'$r$ (cm)')
plt.ylabel(r'cm^2/s^2')

plt.legend()
plt.show()


# ── Dissipation rate plot ───────────

fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)

C2 = 2

eps_11 = ( (D_11/C2)**(3/2) ) / r1
eps_11_low = ( (D_11_low/C2)**(3/2) ) / r1
eps_11_hi = ( (D_11_hi/C2)**(3/2) ) / r1

eps_33 = ( (D_33/C2)**(3/2) ) / r3
eps_33_low = ( (D_33_low/C2)**(3/2) ) / r3
eps_33_hi = ( (D_33_hi/C2)**(3/2) ) / r3

# eps_13 = ( ((3*D_13)/(4*C2))**(3/2) ) / r3
# eps_31 = ( ((3*D_31)/(4*C2))**(3/2) ) / r1

axes[0].semilogx(r1, eps_11, label='e_11', c='r')
axes[0].semilogx(r1, eps_11_low, label='CI', c='black', linestyle='--')
axes[0].semilogx(r1, eps_11_hi, label='CI', c='black', linestyle='--')

axes[1].semilogx(r3, eps_33, label='e_33', c='b')
axes[1].semilogx(r3, eps_33_low, label='CI', c='black', linestyle='--')
axes[1].semilogx(r3, eps_33_hi, label='CI', c='black', linestyle='--')

# plt.axvspan(r1[max_beg11], r1[max_end11], alpha=0.2, color='r', label='inertial range (D_11)')
# plt.axvspan(r3[max_beg33], r1[max_end33], alpha=0.2, color='b', label='inertial range (D_33)')

axes[0].axhline(np.mean(eps_11[max_beg11:max_end11]), color='r', linestyle='--', label=f'eps = {np.mean(eps_11[max_beg11:max_end11])}')
axes[1].axhline(np.mean(eps_33[max_beg33:max_end33]), color='b', linestyle='--', label=f'eps = {np.mean(eps_33[max_beg33:max_end33])}')

axes[0].set_xlabel(r'$r$ (cm)')
axes[0].set_ylabel(r'$cm^2/s^3$')
axes[1].set_xlabel(r'$r$ (cm)')
axes[0].legend()
axes[1].legend()
plt.show()

print(f'eps_11, eps_33 = {np.mean(eps_11[max_beg11:max_end11])}, {np.mean(eps_33[max_beg33:max_end33])}')


# ── Integral length plot ───────────

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

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

L_fit = np.trapezoid(rho_fit, r_fit)

r_ext = np.linspace(min(rj[1:]), 10*Lm_fit, 1500)  # extend a few length scales
rho_ext = model(r_ext, q_fit, Lm_fit)
L_ext = np.trapezoid(rho_ext, r_ext)

idx = np.argmax(rj)
first_zero = np.argmax(rho_ij < 0)
if first_zero > 0:
    print(first_zero)
    rho_ij = rho_ij[:first_zero]
    rho_ij_low = rho_ij_low[:first_zero]
    rho_ij_hi = rho_ij_hi[:first_zero]
    rj = rj[:first_zero]
    L_data = np.trapezoid(rho_ij, rj)
L_data = np.trapezoid(rho_ij, rj)

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

# print(f'min r = {min(r1)}')
r_fit = np.linspace(min(rj[1:]), max(rj), 500)
rho_fit = model(r_fit, q_fit, Lm_fit)

L_data = np.trapezoid(rho_ij, rj)
L_fit = np.trapezoid(rho_fit, r_fit)

r_ext = np.linspace(min(rj[1:]), 10*Lm_fit, 1500)  # extend a few length scales
rho_ext = model(r_ext, q_fit, Lm_fit)
L_ext = np.trapezoid(rho_ext, r_ext)

idx = np.argmax(rj)
first_zero = np.argmax(rho_ij < 0)
if first_zero > 0:
    print(first_zero)
    rho_ij = rho_ij[:first_zero]
    rho_ij_low = rho_ij_low[:first_zero]
    rho_ij_hi = rho_ij_hi[:first_zero]
    rj = rj[:first_zero]

L_data = np.trapezoid(rho_ij, rj)
L_data_low = np.trapezoid(rho_ij_low, rj)
L_data_hi = np.trapezoid(rho_ij_hi, rj)

axes[1].plot(rj, rho_ij, label=f'Data: L = {L_data} CI= {L_data_low, L_data_hi}')
axes[1].plot(r_fit, rho_fit, 'r', label=f'Model fit: L = {L_fit}')
axes[1].plot(rj, rho_ij_low, label=f'CI', linestyle='--')
axes[1].plot(rj, rho_ij_hi, label=f'CI', linestyle='--')
axes[1].plot(r_ext[len(r_fit):], rho_ext[len(r_fit):], 'r', label=f'Extended model: L = {L_ext}', linestyle='--')
axes[1].set_xlabel(r'r (cm)')
axes[1].legend()
plt.show()


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


