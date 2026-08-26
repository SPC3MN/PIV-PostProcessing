import os
import glob
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter



# Single-column plot	(3.4, 2.5–3.0)
# Two stacked subplots	(3.4, 5.5)
# Double-column plot	(7.0, 3.5–5.0)


plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})


def load_single(file, Ny, Nx):
    print(f"\r{os.path.basename(file)}", end = "")
    df = pd.read_csv(file, sep=';', usecols=['x [mm]', 'y [mm]', 'Velocity u [m/s]', 'Velocity v [m/s]'])
    U = df.pivot_table(index='y [mm]', columns='x [mm]', values='Velocity u [m/s]').values
    V = df.pivot_table(index='y [mm]', columns='x [mm]', values='Velocity v [m/s]').values

    return U[:Ny, :Nx], V[:Ny, :Nx]


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

    U_all, V_all = zip(*results)

    print("\n" + f"Loading done: {round(time.perf_counter() - start, 3)} s" + "\n")

    return  X, Y, np.stack(U_all), np.stack(V_all)


# --------------------------------------------
# Control
# --------------------------------------------
def Center_Contour(X, Y, U, V, width=300, height=250):
    x_center = X[0, :].mean()
    y_center = Y[:, 0].mean()

    xmask = np.abs(X[0, :] - x_center) <= width / 2
    ymask = np.abs(Y[:, 0] - y_center) <= height / 2

    idx = np.ix_(ymask, xmask)

    X_c = X[idx] - x_center
    Y_c = Y[idx] - y_center

    return X_c, Y_c, U[idx], V[idx]
Load = True
input_dir = '/Volumes/PIV Data1/Final_Single_Jet/'

cases = ['straight', '8-10-20-FINAL']

f = cases


 # 0, 0.656, 1.3125, 2.625, 5.25

cycle_len = 24 # seconds

if Load:

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 3))

    for idx, case in enumerate(cases):

        stat_file = '/Volumes/PIV Data1/Single_Jet_Processed/' + case + '/Statistics.npz'
        d = np.load(stat_file)

        Y_c = d['Y_centered']

        # KE_total = d['KE_total']
        # Ens_total = d['Ens_total']
        U_profile = d['U_profile']
        print(case, np.nanmax(U_profile))
        mask = U_profile <= 0.5 * np.max(U_profile)

        maskedY = np.ma.array(Y_c, mask=mask)
        upper = np.max(maskedY, axis=0)
        lower = np.min(maskedY, axis=0)

        half_width = (abs(upper) + abs(lower)) / 2

        print(half_width)

        spreading_rate = d['spreading_rate']
        #print(f[idx], spreading_rate)
        # profiles = d['profiles']

        # for profile in profiles:
        #     plt.plot(Y_c, profile)
        # plt.show()

        # t = np.linspace(0, cycle_len, len(KE_total))
        #
        axes[0].plot(Y_c, gaussian_filter(U_profile-np.min(U_profile), 3), label=f'f = {case}')
        axes[0].set_ylabel(r'$\langle u \rangle$')
        axes[0].set_xlabel(r'frame height (mm)')

        axes[1].scatter(f[idx], spreading_rate, label=f'f = {case}')
        axes[1].set_ylabel(r'Spreading Rate $dr_{1/2}/dx$')
        # axes[1].set_xlabel(r'Burst Fraction $(f)$')

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center right')

    # Make room on the right
    fig.tight_layout()

plt.show()
# plt.savefig("figure.pdf", bbox_inches="tight")