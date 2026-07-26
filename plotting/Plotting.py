import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from fractions import Fraction


# file = 'Planar_Results.csv'
file = 'Planar_Results.csv'
# columns = ['t_on', 't_burst', 'f', 'Umean', 'Vmean', 'Urms', 'Vrms', 'IR', 'M1', 'TKE', 'TKE_dev', 'eps11', 'eps33', 'L11', 'L33', 'IR2', 'lambda1', 'lambda3']
columns = [ "t_on","N_bursts","t_burst","f","Umean","Vmean",
            "Urms","Vrms","IR","M1","TKE","TKE_spatial","eps11",
            "eps33", "L11", "L33", "IR2","lambda1","lambda3","Re_L","C_eps_11","C_eps_33",
]
df = pd.read_csv(file, usecols = columns)

# t_on, t_burst, f, Umean, Vmean, Urms, Vrms, IR, M1, TKE, TKE_dev, eps11, eps33, L11, L33, IR2, lambda1, lambda3 = [np.array(df[col].tolist()) for col in df.columns]

t_on, N_bursts, t_burst, f, Umean, Vmean, Urms, Vrms, IR, M1, TKE, TKE_spatial, eps11, eps33, L11, L33, IR2, lambda1, lambda3, Re_L, C_eps_11, C_eps_33 = [np.array(df[col].tolist()) for col in df.columns]

t_on_groups = {k: v for k, v in df.groupby("t_on")}
t_burst_groups = {k: v for k, v in df.groupby("t_burst")}
f_groups = {k: v for k, v in df.groupby("f")}

vu = 1e-6

for value in t_burst_groups:
    plt.scatter(t_burst_groups[value]['TKE'], t_burst_groups[value]['IR'], label=value)

plt.legend()
plt.show()

