"""Standalone: Lumley anisotropy invariant maps + triangle plot for stereo
cases.

Loads only Lumley_Statistics.npz per case (written by Stereo_Decomposition.py
when Save_Lumley is enabled).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.discovery import discover_case_dirs_or_root
from common.prompts import ask_text

# --------------------------------------------
# Control
# --------------------------------------------
processed_root = input(
    "Enter the processed-results directory (Stereo_Decomposition.py output; "
    "a single case folder, or a parent of many): "
).strip()
only = ask_text("Limit to one case name (blank = process every case found)")

case_dirs = discover_case_dirs_or_root(
    processed_root, required_glob=os.path.join("Lumley_Statistics", "Lumley_Statistics.npz"))
if not case_dirs:
    raise FileNotFoundError(
        f"No Lumley_Statistics/Lumley_Statistics.npz found directly in {processed_root!r} "
        "or in its immediate subfolders.")
if only:
    case_dirs = {only: case_dirs[only]}

print(f"Found {len(case_dirs)} case(s): {', '.join(case_dirs)}")

for case_name, case_dir in case_dirs.items():
    print(f"\n===== {case_name} =====")

    lumley_file = os.path.join(case_dir, "Lumley_Statistics", "Lumley_Statistics.npz")
    d = np.load(lumley_file)

    eta2 = d['eta2']
    xi = d['xi']

    print(f'eta2 mean = {np.nanmean(eta2)}   xi mean = {np.nanmean(xi)}')

    plt.contourf(eta2, levels=10)
    plt.title(f'{case_name}: eta^2')
    plt.colorbar()
    plt.show()

    plt.contourf(xi, levels=10)
    plt.title(f'{case_name}: xi')
    plt.colorbar()
    plt.show()

    # Corner ξ values from boundary intersections
    xi_B = 0.6           # right axisymmetric meets 2-D limit
    xi_C = -0.303         # left axisymmetric meets 2-D limit

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
    plt.title(f'{case_name}: Lumley Triangle')
    plt.tight_layout()
    plt.show()
