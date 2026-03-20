from cProfile import label

import matplotlib.pyplot as plt
from tqdm import tqdm

from Magnetic_Field_Functions_support import longitudinal_magnetic_field_landstreet
from Magnetic_field_measurement import polar_spectrum_star
from Synthetic_spectrum import SyntheticSpectrum, read_vald_mask, degrade_resolution
import numpy as np
import pandas as pd


def create_magnetic_curve(polar_magnetic_field, star_i_rad, star_beta_rad, num_phases, random_phase=True,
                          need_zero_phase=True):
    # Общая настройка для магнитной кривой

    quad_field = 0.0
    octo_field = 0.0
    ad = 0.0

    # Учет нулевой фазы

    if need_zero_phase:
        phase_zero_rad = np.random.uniform(0, 1.0)
    else:
        phase_zero_rad = 0.0

    # Случайная фаза или нет

    if random_phase:
        phase_rad = np.random.uniform(0, 1.0, num_phases) + phase_zero_rad
    else:
        phase_rad = np.linspace(0, 1.0, num_phases) + phase_zero_rad

    long_b = longitudinal_magnetic_field_landstreet(phase_rad, star_i_rad, star_beta_rad, polar_magnetic_field, ad,
                                                    quad_field, octo_field)

    return phase_rad, long_b, phase_zero_rad


def synt_polar_spectrum_measure_spectrum(long_magnetic_star, r, vel_rot, signal_noise):
    wavelengths = np.arange(4400.5, 4899.5, 0.001)

    # Чтение маски VALD3
    lines = read_vald_mask('star_t9000_g4_2l4400_4900.lin', wavelengths)

    # Создание синтетического спектра
    synth = SyntheticSpectrum(lines, snr=signal_noise)

    wavelengths, flux_L, flux_R = synth.spectrum_with_magnetic(wavelengths, B_field=long_magnetic_star, vsini=vel_rot)

    wavelengths_res, flux_L = degrade_resolution(wavelengths, flux_L, r)
    _, flux_R = degrade_resolution(wavelengths, flux_R, r)

    wavelengths = wavelengths_res

    # Пример: сохранить маску в CSV
    pd.DataFrame([vars(line) for line in lines]).rename(columns={'wavelength': 'lambda', 'C': 'depth'}).to_csv(
        'test_stars_mask.csv', index=False)

    Spectrum_mask = 'test_stars_mask.csv'
    vsini = vel_rot
    star_name = 'test_star'

    line_parameter = pd.read_csv(Spectrum_mask, sep=',')

    star = polar_spectrum_star(wavelengths, flux_L, flux_R, line_parameter, -80.0, 75.0, vsini, star_name)

    measure_mf, error_measure_mf = star.compute_magnetic_field_by_method('DM_whole')

    return measure_mf, error_measure_mf


def create_one_star(polar_magnetic_field, i_rad, beta_rad, n_phi, resolution, vel_rot, snr, ran_phase=True,
                    zero_phase=True):
    phases, l_magnetic_fields, phase_0 = create_magnetic_curve(polar_magnetic_field, i_rad, beta_rad, n_phi,
                                                               ran_phase, zero_phase)

    lmf_measure = []
    lmf_measure_error = []

    for lmf in tqdm(l_magnetic_fields):
        lb, err_lb = synt_polar_spectrum_measure_spectrum(lmf, resolution, vel_rot, snr)
        lmf_measure.append(lb)
        lmf_measure_error.append(err_lb)

    synt_observ = pd.DataFrame({'phase': phases - phase_0, '<B_l>': lmf_measure, '<B_err>': lmf_measure_error})

    synt_observ.to_csv('Test_synt_data.csv', index=False)

    return synt_observ, phase_0


if __name__ == '__main__':
    print('Test')

    # Задается звезда параметры ее поля

    b_p0 = 8.0E+2
    decline_rotation_rad = np.random.uniform(0, np.pi)
    decline_polar_rad = np.random.uniform(0, np.pi)

    num_phi = 5
    vel_sin_i = 75.0

    spectrum_resol = 10000.0
    spectrum_signal_noise = 100.0

    rand_phi = True
    need_phi_0 = True

    data, f_0 = create_one_star(b_p0, decline_rotation_rad, decline_polar_rad, num_phi, spectrum_resol, vel_sin_i, spectrum_signal_noise, rand_phi, need_phi_0)

    true_phase = np.linspace(0, 1.0, 100) + f_0

    print('Statistick data')

    print(f'Decline rotation: {decline_rotation_rad * 180.0 / np.pi: .2f} degrees')
    print(f'Decline magnetic polar: {decline_polar_rad * 180.0 / np.pi: .2f} degrees')
    print(f'Phase 0: {f_0 * 2.0 * np.pi * 180.0 / np.pi: .2f} degrees')

    print(f'Polar Magnetic Field: {b_p0}')

    print(f'Average quad magnetic field: {np.sqrt(np.mean(np.square(data['<B_l>'])))} +- {np.sqrt(np.mean(np.square(data['<B_err>'])))}')

    with open('stat_output.txt', 'w') as f:
        f.write('Statistick data\n')

        f.write(f'Decline rotation: {decline_rotation_rad * 180.0 / np.pi: .2f} degrees\n')
        f.write(f'Decline magnetic polar: {decline_polar_rad * 180.0 / np.pi: .2f} degrees\n')
        f.write(f'Phase 0: {f_0 * 2.0 * np.pi * 180.0 / np.pi: .2f} degrees\n')

        f.write(f'Polar Magnetic Field: {b_p0}\n')

        f.write(
            f'Average quad magnetic field: {np.sqrt(np.mean(np.square(data['<B_l>']))): .2f} +- {np.sqrt(np.mean(np.square(data['<B_err>']))): .2f}')

    plt.errorbar(data['phase'], data['<B_l>'], yerr=data['<B_err>'], fmt='.', capsize=2, label='<B_l>')
    plt.plot(true_phase - f_0, longitudinal_magnetic_field_landstreet(true_phase, decline_rotation_rad, decline_polar_rad, b_p0,0.0, 0.0, 0.0), label='True')
    plt.legend()
    plt.show()