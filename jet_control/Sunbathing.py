import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random


def plot_pump_signals(pump_control, dt=0.1):
    """
    Plot all pump signals as a raster plot.
    """
    num_pumps, num_steps = pump_control.shape
    t = np.arange(num_steps) * dt  # time vector

    plt.figure(figsize=(12, 6))
    plt.imshow(pump_control, aspect='auto', cmap='Greys', origin='lower',
               interpolation='none', extent=[0, t[-1], 0, num_pumps])

    plt.xlabel("Time [s]")
    plt.ylabel("Pump index")
    plt.title("Pump ON/OFF signals (0=ON, 1=OFF)")
    plt.colorbar(label="State (0=ON, 1=OFF)")
    plt.show()


def Sunbathing_signal(m_on, source_fraction, t_on, f1, f2, f3, dt=0.1):
    """
    Generate one burst ON/OFF cycle as a discrete signal (0=ON, 1=OFF)
    """
    print(int(round(t_on/dt)))
    print(int(round(m_on / dt)))
    mu_on, t_bon = int(round(m_on/dt)), int(round(t_on/dt))

    rng = np.random.default_rng(None)

    # Sample ON time
    T_on = abs(int(rng.normal(mu_on, 0)))

    # Compute OFF time from source fraction
    mu_off = mu_on * (1 - source_fraction) / source_fraction
    T_off = abs(int(rng.normal(mu_off, mu_off/3)))


    # Number of bursts
    Nb = int(T_on / t_bon)

    t_boff1 = int(round(f1/dt))
    t_boff2 = int(round(f2/dt))
    t_boff3 = int(round(f3/dt))

    T_off1 = T_off2 = T_off

    while T_off < (Nb)*t_boff1 or T_off < (Nb)*t_boff2 or T_off < (Nb)*t_boff3:
        print(f'edge_case: t_boff = {t_boff1}, {t_boff2}')

        T_off2 = abs(int(rng.normal(mu_off, mu_off / 3)))
        T_off = T_off2


    # Calculate an ordinary sunbathing signal
    S = []
    on = [0] * T_on
    off = [1] * T_off
    S.extend(on)
    S.extend(off)


    # Calculate a pulsed sunbathing signal with psi 1
    B1_signal = []
    T_counter = T_off
    T_counter_on = T_on

    for _ in range(Nb - 1):
        on = [0] * t_bon
        off = [1] * t_boff1
        B1_signal.extend(on)
        B1_signal.extend(off)
        T_counter -= np.sum(off)
        T_counter_on -= t_bon

    B1 = [1] * int(T_counter)
    B1_signal.extend([0] * T_counter_on)
    B1_signal.extend(B1)

    # Calculate a pulsed sunbathing signal with psi 2
    B2_signal = []
    T_counter = T_off
    T_counter_on = T_on

    for _ in range(Nb-1):

        on = [0] * t_bon
        off = [1]*t_boff2
        B2_signal.extend(on)
        B2_signal.extend(off)
        T_counter -= np.sum(off)
        T_counter_on -= t_bon

    B2 = [1]*int(T_counter)
    B2_signal.extend([0]*T_counter_on)
    B2_signal.extend(B2)

    # Calculate a pulsed sunbathing signal with psi 3
    B3_signal = []
    T_counter = T_off
    T_counter_on = T_on

    for _ in range(Nb-1):

        on = [0] * t_bon
        off = [1]*t_boff3
        B3_signal.extend(on)
        B3_signal.extend(off)
        T_counter -= np.sum(off)
        T_counter_on -= t_bon

    B3 = [1]*int(T_counter)
    B3_signal.extend([0]*T_counter_on)
    B3_signal.extend(B3)

    # print(S.count(1), B1_signal.count(1), B2_signal.count(1), B3_signal.count(1))



    return S, B1_signal, B2_signal, B3_signal, t_boff1, t_boff2, t_boff3

def generate_pump_control(num_pumps, test_duration, mu_on, source_fraction, t_bon, f1, f2, f3, dt=0.1):
    """
    Generate control signals for multiple pumps over a total test duration
    """
    num_steps = int(test_duration / dt)
    S_full = np.ones((num_pumps, num_steps), dtype=int)  # start OFF
    B1_full = np.ones((num_pumps, num_steps), dtype=int)  # start OFF
    B2_full = np.ones((num_pumps, num_steps), dtype=int)  # start OFF
    B3_full = np.ones((num_pumps, num_steps), dtype=int)  # start OFF

    off_distribution1 = []
    off_distribution2 = []
    off_distribution3 = []
    for pump in range(num_pumps):
        t_index = 0
        while t_index < num_steps:
            S, B1, B2, B3, T_off1, T_off2, T_off3 = Sunbathing_signal(mu_on, source_fraction, t_bon, f1, f2, f3, dt=0.1)

            off_distribution1.append(T_off1/10)
            off_distribution2.append(T_off2/10)
            off_distribution3.append(T_off3/10)

            cycle_len = len(B1)


            # Fit the cycle into remaining time and add a 60s buffer at start and 60s buffer at the end
            if t_index + cycle_len > num_steps:

                break

            S_full[pump, t_index:t_index + cycle_len] = S
            B1_full[pump, t_index:t_index + cycle_len] = B1
            B2_full[pump, t_index:t_index + cycle_len] = B2
            B3_full[pump, t_index:t_index + cycle_len] = B3

            t_index += cycle_len

        N = 600
        S_full[pump][:N] = 1
        B1_full[pump][:N] = 1
        B2_full[pump][:N] = 1
        B3_full[pump][:N] = 1

    return S_full, B1_full, B2_full, B3_full, off_distribution1, off_distribution2, off_distribution3

pump_control, B1_control, B2_control, B3_control, off_distribution1, off_distribution2, off_distribution3 = generate_pump_control(
    num_pumps=96,
    test_duration=2280,  # total time in seconds
    mu_on=6,  # mean ON time
    source_fraction=0.125,  # duty cycle
    t_bon=0.7,  # ON segment of burst
    f1=0.3, # Off segment of burst
    f2=0.7,
    f3=1.5,
    dt=0.1  # timestep
)






print(f'mean off burst length 1 = {np.mean(off_distribution1)}+-{np.std(off_distribution1)}, maxmin = {np.max(off_distribution1), np.min(off_distribution1)}')
print(f'mean off burst length 2 = {np.mean(off_distribution2)}+-{np.std(off_distribution2)}, maxmin = {np.max(off_distribution2), np.min(off_distribution2)}')
print(f'mean off burst length 3 = {np.mean(off_distribution3)}+-{np.std(off_distribution3)}, maxmin = {np.max(off_distribution3), np.min(off_distribution3)}')



print(len(pump_control[0]))

# data_to_save = pump_control.T  # shape = (timesteps, pumps)
# np.savetxt("6s.txt", data_to_save, fmt='%i', delimiter=',')

data_to_save = B1_control.T  # shape = (timesteps, pumps)
np.savetxt("6s-0_7-0_3.txt", data_to_save, fmt='%i', delimiter=',')

data_to_save = B2_control.T  # shape = (timesteps, pumps)
np.savetxt("6s-0_7-0_7.txt", data_to_save, fmt='%i', delimiter=',')
#
data_to_save = B3_control.T  # shape = (timesteps, pumps)
np.savetxt("6s-0_7-1_5.txt", data_to_save, fmt='%i', delimiter=',')




data_list = [pump_control, B1_control, B2_control, B3_control]
for data in data_list:
    # # Transpose so each column is a pump
    data_to_save = data.T  # shape = (timesteps, pumps)
    # Row sum = number of OFF pumps
    row_sums = np.sum(data_to_save, axis=1)
    # Maximum pumps ON at any timestep
    max_pumps_on = data_to_save.shape[1] - np.min(row_sums)
    print(max_pumps_on)
    # plot_pump_signals(data, dt=0.1)
