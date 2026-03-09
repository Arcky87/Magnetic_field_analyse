import matplotlib.pyplot as plt

from Magnetic_Field_Functions_support import average_magnetic_field, average_magnetic_field_longitudinal
import numpy as np
import pandas as pd

def create_one_random_star(polar_magnetic_field, level_error_magnetic_field, num_phases, random_phase=False, need_zero_phase=False):
    star_i_rad = np.random.uniform(0, np.pi)
    star_beta_rad = np.random.uniform(0, np.pi)

    if need_zero_phase:
        phase_zero_rad = np.random.uniform(0, 2.0 * np.pi)
    else:
        phase_zero_rad = 0.0

    if random_phase:
        phase_rad = np.random.uniform(0, 2.0 * np.pi, num_phases) + phase_zero_rad
    else:
        phase_rad = np.linspace(0, 2.0 * np.pi, num_phases) + phase_zero_rad

    return

if __name__ == '__main__':
    print('Test')
