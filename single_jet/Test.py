import os
import glob
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter





def load_single(file, top, bottom, left, right):
    print(f"\r{os.path.basename(file)}", end = "")
    df = pd.read_csv(file, sep=';', usecols=['x [mm]', 'y [mm]', 'Velocity x [m/s]', 'Velocity y [m/s]'])

    U = df.pivot_table(index='y [mm]', columns='x [mm]', values='Velocity x [m/s]').values
    V = df.pivot_table(index='y [mm]', columns='x [mm]', values='Velocity y [m/s]').values
    # print((100 * np.isnan(U[:Ny, :Nx]).sum()) / U[:Ny, :Nx].size, (100 * np.isnan(U[:Ny, :Nx]).sum())/ U[:Ny, :Nx].size)
    return U[top:bottom, left:right], V[top:bottom, left:right]


def load_dataset(csv_dir, cutoff, width_mm, height_mm):

    if cutoff:
        csv_files = sorted(
            f for f in glob.glob(os.path.join(csv_dir, '*.csv')) if not os.path.basename(f).startswith('._'))[:cutoff]
    else:
        csv_files = sorted(
            f for f in glob.glob(os.path.join(csv_dir, '*.csv')) if not os.path.basename(f).startswith('._'))

    print(f'Loading {len(csv_files)} Files... ')
    df0 = pd.read_csv(csv_files[0], sep=';')
    U0 = df0.pivot_table(index='y [mm]', columns='x [mm]', values='Velocity x [m/s]')

    Ny0, Nx0 = U0.shape
    print(f'Original Size: {(Ny0, Nx0)}')

    # Coordinates (full FOV, untrimmed)
    x_coords_full = U0.columns.values
    y_coords_full = U0.index.values

    # Grid spacing
    dx = np.median(np.diff(x_coords_full))
    dy = np.median(np.diff(y_coords_full))
    print(f'dx = {dx:.4f} mm, dy = {dy:.4f} mm')

    # Actual physical center of the FOV (NOT assumed to be 0,0)
    x_center = (x_coords_full.min() + x_coords_full.max()) / 2
    y_center = (y_coords_full.min() + y_coords_full.max()) / 2
    print(f'FOV center: x={x_center:.3f} mm, y={y_center:.3f} mm')

    # Desired number of points to cover width_mm / height_mm
    Nx = int(round(width_mm / dx))
    Ny = int(round(height_mm / dy))

    # Find the index of the coordinate closest to the FOV center
    x_center_idx = np.argmin(np.abs(x_coords_full - x_center))
    y_center_idx = np.argmin(np.abs(y_coords_full - y_center))

    # Build symmetric index window around the center index
    left = x_center_idx - Nx // 2
    right = left + Nx
    top = y_center_idx - Ny // 2
    bottom = top + Ny

    # Clip to valid range (in case desired size exceeds available FOV)
    if left < 0 or top < 0 or right > Nx0 or bottom > Ny0:
        raise ValueError(
            f"Requested frame ({width_mm} x {height_mm} mm) exceeds available FOV "
            f"({Nx0*dx:.1f} x {Ny0*dy:.1f} mm). "
            f"Computed indices: left={left}, right={right}, top={top}, bottom={bottom}"
        )

    print(f'Trimmed to: {(Ny, Nx)} points -> {Nx*dx:.2f} x {Ny*dy:.2f} mm, '
          f'centered at ({x_center:.2f}, {y_center:.2f}) mm')

    X, Y = np.meshgrid(x_coords_full, y_coords_full)
    X = X[top:bottom, left:right]
    Y = Y[top:bottom, left:right]

    with ThreadPoolExecutor() as ex:
        results = list(ex.map(lambda f: load_single(f, top, bottom, left, right), csv_files))

    # filter out empty results (any array with shape (0, 0))
    results = [
        r for r in results
        if all(arr.shape != (0, 0) for arr in r)
    ]

    U_all, V_all = zip(*results)

    print("\n" + f"Loading done: {round(time.perf_counter() - start, 3)} s" + "\n")

    return X, Y, np.stack(U_all), np.stack(V_all)
# --------------------------------------------
# Animation function
# --------------------------------------------
def animate(X, Y, U_all, V_all, interval):


    Nt = len(U_all)

    fig, ax = plt.subplots(
        figsize=(14, 5),
        constrained_layout=True
    )

    # ----------------------------------------
    # Plot helper
    # ----------------------------------------
    def plot(ax, X, Y, U, V, title):
        X, Y, U, V = Center_Contour(X, Y, U, V)
        U = gaussian_filter(U, 1)
        V = gaussian_filter(V, 1)
        ax.clear()
        ax.contourf(X, Y, np.sqrt(U**2 + V**2), levels=10, vmin=0.25, vmax=0.45)
        ax.set_title(title)

    # ----------------------------------------
    # Animation update
    # ----------------------------------------
    def update(frame):

        plot(
            ax,
            X, Y,
            U_all[frame],
            V_all[frame],
            f'Dataset 1 | Frame {frame}'
        )


    # ----------------------------------------
    # Create animation
    # ----------------------------------------
    anim = FuncAnimation(
        fig,
        update,
        frames=Nt,
        interval=interval
    )

    return anim
def Half_Width(X, Y, U_all):

    U_mean = np.nanmean(U_all, axis=0)
    X, Y, U_mean, _ = Center_Contour(X, Y, U_mean, U_mean)


    U_mean = gaussian_filter(U_mean, 3)
    mask = U_mean <= 0.5 * np.max(U_mean, axis=0)

    maskedY = np.ma.array(Y, mask=mask)
    upper = np.max(maskedY, axis=0)
    lower = np.min(maskedY, axis=0)

    half_width = (abs(upper) + abs(lower) ) / 2

    print(f'half width: {np.nanmax(half_width), np.nanmean(half_width), np.nanmin(half_width)}')

    m, b = np.polyfit(X[0], half_width, 1)

    return m
def KE_total(X, Y, U_all, V_all):

    KE_list = []
    KE = 0
    for U, V in zip(U_all, V_all):
        X, Y, U, V = Center_Contour(X, Y, U, V)
        KE += np.nanmean(U ** 2 + V ** 2) / 2
        KE_list.append(KE)

    return KE_list
def Velocity_Profile(X, Y, U_all):
    U_centered = []
    for U in U_all:
        Xc, Yc, U_c, _ = Center_Contour(X, Y, U, U)
        U_centered.append(U_c)
    profile = np.nanmean(np.array(U_centered), axis=(0, 2))
    Y_mid = Yc[:, 0][np.argmax(profile)]
    Y_c = Yc[:, 0] - Y_mid
    return profile, Y_c
def Profile_Development(X, Y, U_all, cycle_len, dt=40):
    U_centered = []
    for U in U_all:
        Xc, Yc, U_c, _ = Center_Contour(X, Y, U, U)
        U_centered.append(U_c)
    mean_profile = np.nanmean(np.array(U_centered), axis=(0, 2))
    Y_mid = Yc[:, 0][np.argmax(mean_profile)]
    Y_c = Yc[:, 0] - Y_mid

    profile = np.nanmean(np.array(U_centered), axis=2)
    profile_snap = []
    step = dt
    for i in range(int(len(profile)//dt)):
        profile_snap.append(np.nanmean(profile[:step], axis=0))
        step += dt

    return profile_snap

# --------------------------------------------
# Control
# --------------------------------------------
def Center_Contour(X, Y, U, V, width=200, height=175):
    x_center = X[0, :].mean()
    y_center = Y[:, 0].mean()

    xmask = np.abs(X[0, :] - x_center) <= width / 2
    ymask = np.abs(Y[:, 0] - y_center) <= height / 2

    idx = np.ix_(ymask, xmask)

    X_c = X[idx] - x_center
    Y_c = Y[idx] - y_center

    return X_c, Y_c, U[idx], V[idx]
Save = True
Load = False
input_dir = '//Volumes/PIV Data1/Educational/'
cases = ['straight_avg', '8-10']
cutoff_idx = False # False if none
width = 300
height = 200

input = input_dir + cases[0]

start = time.perf_counter()
X, Y, U_all, V_all = load_dataset(input, cutoff=cutoff_idx, width_mm=width, height_mm=height)

plt.contourf(X, Y, np.mean(U_all, axis=0), levels=5, vmin=0, vmax=0.6)
print(np.max(np.mean(U_all, axis=0)), np.min(np.mean(U_all, axis=0)))
plt.show()

spread = Half_Width(X, Y, U_all)
print(spread)