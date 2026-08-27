import os
import glob
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def _centered_crop_bounds(X_full, Y_full, width_mm, height_mm):
    Ny0, Nx0 = X_full.shape

    x_coords_full = X_full[0]
    y_coords_full = Y_full[:, 0]

    dx = np.median(np.diff(x_coords_full))
    dy = np.median(np.diff(y_coords_full))
    print(f'dx = {dx:.4f} mm, dy = {dy:.4f} mm')
    print(f'Full FOV: {(Ny0, Nx0)} points -> {Nx0*abs(dx):.2f} x {Ny0*abs(dy):.2f} mm')

    x_center = (x_coords_full.min() + x_coords_full.max()) / 2
    y_center = (y_coords_full.min() + y_coords_full.max()) / 2
    print(f'FOV center: x={x_center:.3f} mm, y={y_center:.3f} mm')

    Nx = int(round(width_mm / abs(dx)))
    Ny = int(round(height_mm / abs(dy)))

    x_center_idx = np.argmin(np.abs(x_coords_full - x_center))
    y_center_idx = np.argmin(np.abs(y_coords_full - y_center))

    left = x_center_idx - Nx // 2
    right = left + Nx
    top = y_center_idx - Ny // 2
    bottom = top + Ny

    if left < 0 or top < 0 or right > Nx0 or bottom > Ny0:
        raise ValueError(
            f"Requested frame ({width_mm} x {height_mm} mm) exceeds available FOV "
            f"({Nx0*abs(dx):.1f} x {Ny0*abs(dy):.1f} mm). "
            f"Computed indices: left={left}, right={right}, top={top}, bottom={bottom}"
        )

    print(f'Trimmed to: {(Ny, Nx)} points -> {Nx*abs(dx):.2f} x {Ny*abs(dy):.2f} mm, '
          f'centered at ({x_center:.2f}, {y_center:.2f}) mm')
    print(f'NOTE: this discards a buffer around the edges of the full FOV -- '
          f'{left} px left, {Nx0 - right} px right, {top} px top, {Ny0 - bottom} px bottom '
          f'({left*abs(dx):.1f}/{(Nx0-right)*abs(dx):.1f}/{top*abs(dy):.1f}/{(Ny0-bottom)*abs(dy):.1f} mm) removed')

    return top, bottom, left, right


def _resolve_key(data, name):
    """Look up `name` in an open npz archive, tolerating case differences
    between producers (e.g. PIV_GUI writes lowercase 'x'/'y'/'u'/'v', this
    pipeline's own exports use uppercase 'X'/'Y'/'U'/'V')."""
    if name in data.files:
        return name
    lower_map = {k.lower(): k for k in data.files}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    raise KeyError(
        f"{name!r} not found (case-insensitively) in npz archive; available keys: {sorted(data.files)}")


def _load_single(file, components, bounds):
    print(f"\r{os.path.basename(file)}", end="")
    with np.load(file) as data:
        arrs = [data[_resolve_key(data, c)] for c in components]
    if bounds is not None:
        top, bottom, left, right = bounds
        arrs = [a[top:bottom, left:right] for a in arrs]
    return tuple(arrs)


def load_dataset_npz(npz_dir, cutoff, components=("U", "V"), crop=None):
    """Load a case's snapshots from its *.npz files (X, Y, plus the
    requested velocity `components`). Key lookup is case-insensitive, so
    files from producers using different casing (e.g. PIV_GUI's lowercase
    'x'/'y'/'u'/'v') load the same as this pipeline's own uppercase exports.

    `crop`, if given, is a (width_mm, height_mm) tuple: each snapshot is
    cropped to a centered window of that size before stacking. Leave it None
    when the npz files were already trimmed at export time.
    """
    npz_files = sorted(
        f for f in glob.glob(os.path.join(npz_dir, '*.npz')) if not os.path.basename(f).startswith('._'))
    if cutoff:
        npz_files = npz_files[:cutoff]

    print(f'Loading {len(npz_files)} Files... ')

    with np.load(npz_files[0]) as data0:
        X_full, Y_full = data0[_resolve_key(data0, 'X')], data0[_resolve_key(data0, 'Y')]

    if crop is not None:
        width_mm, height_mm = crop
        bounds = _centered_crop_bounds(X_full, Y_full, width_mm, height_mm)
        top, bottom, left, right = bounds
        X, Y = X_full[top:bottom, left:right], Y_full[top:bottom, left:right]
    else:
        bounds = None
        X, Y = X_full, Y_full

    start = time.perf_counter()
    with ThreadPoolExecutor() as ex:
        results = list(ex.map(lambda f: _load_single(f, components, bounds), npz_files))

    # filter out empty results (any array with shape (0, 0))
    results = [r for r in results if all(arr.shape != (0, 0) for arr in r)]

    stacked = tuple(np.stack(arrs) for arrs in zip(*results))

    print("\n" + f"Loading done: {round(time.perf_counter() - start, 3)} s" + "\n")

    return (X, Y) + stacked
