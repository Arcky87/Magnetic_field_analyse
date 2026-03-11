import numpy as np
from scipy.integrate import trapezoid
import scipy.sparse as sp
from math import factorial
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from numpy.polynomial.legendre import legfit, legval
from numpy.polynomial.chebyshev import chebfit, chebval
from scipy import signal, integrate, interpolate, stats
from uncertainties import unumpy
from matplotlib.ticker import NullFormatter, ScalarFormatter

speed_of_light = 299792.458
R_0 = 1.3996E-7
K_0 = 4.6686E-13

font_size = 20


def print_list_method_measurement_magnetic_field():
    list_of_method = ['IM_single', 'MIM_whole', 'MIM_single', 'LSD_IM', 'MDM_whole', 'DM_whole', 'MDM_single']

    print(f'List method for measurements: {list_of_method}')

    return list_of_method


def measurements_error_long_magnetic_field_integral_methods(i_stockes, v_stockes, grid_abscissa, i_stockes_err,
                                                            v_stockes_err, g_eff, mean_wavelength):
    tmp = 0
    tmp2 = 0
    tmp3 = 0
    ew = 0
    mv = 0
    e1 = 0
    e2 = 0
    mn = np.mean(grid_abscissa)
    for i in range(len(grid_abscissa)):
        tmp2 = grid_abscissa[i] - mn
        tmp3 = np.sqrt(np.abs(i_stockes[i]))
        if np.abs(i_stockes[i]) >= 0:
            ew = ew + tmp3 * i_stockes[i]
            e1 = tmp3 * tmp3 * i_stockes_err[i] * i_stockes_err[i]
        if np.abs(v_stockes[i]) >= 0:
            mv = mv + tmp3 * tmp2 * v_stockes[i]
            e2 = e2 + np.power(tmp3, 2.0) * np.power(tmp2, 2.0) * v_stockes_err[i] * v_stockes_err[i]
    Bz = mv / ew / g_eff / mean_wavelength / R_0
    eBz = (np.sqrt(e1 * np.power(mv, 2.0) + e2 * np.power(ew, 2.0)) / np.power(ew, 2.0) / g_eff / mean_wavelength / R_0)

    return eBz, Bz


class polar_spectrum_star:
    def __init__(self, grid_wave, i_l_mod_polarization, i_r_mod_polarization, spectrum_profile, min_velocity_profile,
                 max_velocity_profile, vsini, star_name):

        self.long_magnetic_field_by_lsd_im_whole_spectrum = None
        self.long_magnetic_field_by_mdm_whole_spectrum = None
        self.long_magnetic_field_by_mim_whole_spectrum = None
        self.long_magnetic_field_by_dm_whole_spectrum = None

        self.long_magnetic_field_by_im_single_line = {}
        self.long_magnetic_field_by_mim_single_line = {}
        self.long_magnetic_field_by_mdm_single_line = {}

        self.lines_profile = None

        self.modified_smoothed_line_profiles = None
        self.average_line_profile = None

        self.least_square_deconvolution_line_profile = None

        self.star_name = star_name

        self.v_sini = vsini

        self.spectrum_mask = spectrum_profile
        self.grid_wavelength = grid_wave
        self.i_stockes = (i_l_mod_polarization + i_r_mod_polarization) / 2.0
        self.v_stockes = (i_l_mod_polarization - i_r_mod_polarization) / 2.0
        self.signal_to_noise = DER_SNR(self.i_stockes)
        self.spectrum_error = np.ones(len(self.grid_wavelength)) / self.signal_to_noise
        self.min_velocity_for_analyse = min_velocity_profile
        self.max_velocity_for_analyse = max_velocity_profile
        self.num_of_lines = len(spectrum_profile['lambda'])

    def compute_line_profile(self):
        lines_profile = {}
        for i in range(len(self.spectrum_mask['lambda'])):
            wavelength_0 = self.spectrum_mask['lambda'][i]
            lines_profile[str(wavelength_0)] = []
            c_0 = self.spectrum_mask['depth'][i]
            g_factor = self.spectrum_mask['g_factor'][i]
            min_wave_line_profile = wavelength_0 * (1.0 + self.min_velocity_for_analyse / speed_of_light)
            max_wave_line_profile = wavelength_0 * (1.0 + self.max_velocity_for_analyse / speed_of_light)

            index_line_profile = np.where(
                (self.grid_wavelength >= min_wave_line_profile) & (self.grid_wavelength <= max_wave_line_profile))

            grid_wave_line = self.grid_wavelength[index_line_profile]
            grid_vel_line = (grid_wave_line - wavelength_0) / wavelength_0 * speed_of_light
            grid_i_stockes_line = self.i_stockes[index_line_profile]
            grid_v_stockes_line = self.v_stockes[index_line_profile]

            grid_i_stockes_error_line = self.spectrum_error[index_line_profile]
            grid_v_stockes_error_line = self.spectrum_error[index_line_profile]

            lines_profile[str(wavelength_0)].append([c_0, g_factor, wavelength_0])  # 0
            lines_profile[str(wavelength_0)].append(grid_wave_line)  # 1
            lines_profile[str(wavelength_0)].append(grid_vel_line)  # 2
            lines_profile[str(wavelength_0)].append(grid_i_stockes_line)  # 3
            lines_profile[str(wavelength_0)].append(grid_v_stockes_line)  # 4
            lines_profile[str(wavelength_0)].append(grid_i_stockes_error_line)  # 5
            lines_profile[str(wavelength_0)].append(grid_v_stockes_error_line)  # 6

        return lines_profile

    def compute_least_square_deconvolution_line_profile(self):
        spectrum_error = np.ones(len(self.grid_wavelength)) / self.signal_to_noise

        delta_velocity_observation = speed_of_light * np.mean(np.diff(self.grid_wavelength)) / np.mean(
            self.grid_wavelength[0])

        grid_velocity = np.arange(self.min_velocity_for_analyse,
                                  self.max_velocity_for_analyse + delta_velocity_observation,
                                  delta_velocity_observation)

        wavelengths_mask = np.array(list(self.spectrum_mask['lambda']))
        average_wavelengths_mask = np.mean(self.spectrum_mask['lambda'])
        average_depth_mask = np.mean(self.spectrum_mask['depth'])
        average_g_factor = np.mean(self.spectrum_mask['g_factor'])

        weight = np.zeros((2, len(wavelengths_mask)))

        weight[0, :] = self.spectrum_mask['depth'] * self.spectrum_mask['lambda'] / (
                average_depth_mask * average_wavelengths_mask)
        weight[1, :] = self.spectrum_mask['depth'] * self.spectrum_mask['lambda'] * self.spectrum_mask['g_factor'] / (
                average_g_factor * average_depth_mask * average_wavelengths_mask)

        wavelengths_mask_sub = wavelengths_mask[
            (wavelengths_mask >= self.grid_wavelength[0]) & (wavelengths_mask <= self.grid_wavelength[-1])]
        wavelengths_trim = np.ones(len(self.grid_wavelength)) * np.nan

        for line in wavelengths_mask_sub:
            delta_line_min = (self.min_velocity_for_analyse - delta_velocity_observation) * line / speed_of_light
            delta_line_max = (self.max_velocity_for_analyse - delta_velocity_observation) * line / speed_of_light
            index_ss = np.where(
                (self.grid_wavelength >= line + delta_line_min) & (self.grid_wavelength <= line + delta_line_max))
            wavelengths_trim[index_ss] = self.grid_wavelength[index_ss]

        i_stockes = self.i_stockes[~np.isnan(wavelengths_trim)]
        v_stockes = self.v_stockes[~np.isnan(wavelengths_trim)] / i_stockes
        spectrum_error = spectrum_error[~np.isnan(wavelengths_trim)]
        wavelengths_observation = self.grid_wavelength[~np.isnan(wavelengths_trim)]

        matrix_S2 = fill_S_sparse(wavelengths_observation, spectrum_error)

        matrix_M_v_stockes = fill_M_sparse(wavelengths_observation, wavelengths_mask, weight[1, :], grid_velocity)
        vector_Z_v, chisq_v, Zv_errb = solve_least_square_deconvolution_sparse(v_stockes, matrix_S2, matrix_M_v_stockes,
                                                                               'v', 8.0E+4)
        vector_Z_v = np.array(vector_Z_v).ravel()

        matrix_M_i_stockes = fill_M_sparse(wavelengths_observation, wavelengths_mask, weight[0, :], grid_velocity)
        vector_Z_i, chisq_i, Zi_errb = solve_least_square_deconvolution_sparse(i_stockes, matrix_S2, matrix_M_i_stockes,
                                                                               'i', 8.0E+4)
        vector_Z_i = 1.0 - np.array(vector_Z_i).ravel()

        vector_Z_i = renormalize_iraf(grid_velocity, vector_Z_i, "chebyshev", 1, 7, 1.2, 6.0, 5)

        return grid_velocity, vector_Z_i, vector_Z_v, Zi_errb, Zv_errb

    def compute_modified_smoothed_line_profile(self, smooth_parameter_k, lines_profile):
        N = self.num_of_lines
        self.modified_smoothed_line_profiles = {}

        for i in range(N):
            wavelength_0 = self.spectrum_mask['lambda'][i]
            sigma = self.v_sini / speed_of_light * wavelength_0
            g_eff = self.spectrum_mask['g_factor'][i]

            smooth_parameter_S = 0.4 * 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma / wavelength_0 * speed_of_light

            j_smoothed = interpol_spec(lines_profile[str(wavelength_0)][2],
                                       lines_profile[str(wavelength_0)][3],
                                       lines_profile[str(wavelength_0)][2], 1)

            convolution_v = operator_convolution(smooth_parameter_k, lines_profile[str(wavelength_0)][4],
                                                 lines_profile[str(wavelength_0)][2])
            convolution_j = operator_convolution(smooth_parameter_k, j_smoothed,
                                                 lines_profile[str(wavelength_0)][2])

            convolution_error = operator_convolution(smooth_parameter_k, lines_profile[str(wavelength_0)][6],
                                                     lines_profile[str(wavelength_0)][2])

            grid_smooth_v = np.array(
                [convolution_v(vel, smooth_parameter_S) for vel in lines_profile[str(wavelength_0)][2]])
            grid_smooth_j = R_0 * g_eff * wavelength_0 * np.array(
                [convolution_j(vel, smooth_parameter_S) for vel in lines_profile[str(wavelength_0)][2]])

            grid_smooth_error = np.array(
                [convolution_error(vel, smooth_parameter_S) for vel in lines_profile[str(wavelength_0)][2]])

            self.modified_smoothed_line_profiles[str(wavelength_0)] = [grid_smooth_v, grid_smooth_j,
                                                                       grid_smooth_error,
                                                                       R_0 * g_eff * wavelength_0 * grid_smooth_error]

    def compute_average_smoothed_line_profile(self, smoothed_parameter_k, lines_profile):
        if self.modified_smoothed_line_profiles is None:
            self.compute_modified_smoothed_line_profile(smoothed_parameter_k, lines_profile)
        N = self.num_of_lines
        g_line = np.zeros(N)

        average_profile_V = 0.0
        average_profile_J = 0.0
        average_profile_V_error = 0.0
        average_profile_J_error = 0.0

        for i in range(N):
            wavelength_0 = self.spectrum_mask['lambda'][i]
            C_0 = self.spectrum_mask['depth'][i]
            g_eff = self.spectrum_mask['g_factor'][i]
            g_line[i] = C_0 * g_eff * np.power(wavelength_0, 2.0)
        min_velocity_average_line_profile = max(
            [lines_profile[str(self.spectrum_mask['lambda'][i])][2][0] for i in range(N)])
        max_velocity_average_line_profile = min(
            [lines_profile[str(self.spectrum_mask['lambda'][i])][2][-1] for i in range(N)])
        delta_velocity = min([abs(
            lines_profile[str(self.spectrum_mask['lambda'][i])][2][0] -
            lines_profile[str(self.spectrum_mask['lambda'][i])][2][1])
            for i in range(N)])

        grid_velocity_average_profile_line = np.arange(min_velocity_average_line_profile,
                                                       max_velocity_average_line_profile, delta_velocity)

        for i in range(N):
            wavelength_0 = self.spectrum_mask['lambda'][i]
            average_profile_V += g_line[i] * interpol_spec(lines_profile[str(wavelength_0)][2],
                                                           self.modified_smoothed_line_profiles[str(wavelength_0)][0],
                                                           grid_velocity_average_profile_line, 0) / sum(g_line)
            average_profile_J += g_line[i] * interpol_spec(lines_profile[str(wavelength_0)][2],
                                                           self.modified_smoothed_line_profiles[str(wavelength_0)][1],
                                                           grid_velocity_average_profile_line, 0) / sum(g_line)

            average_profile_V_error += g_line[i] * interpol_spec(lines_profile[str(wavelength_0)][2],
                                                                 self.modified_smoothed_line_profiles[
                                                                     str(wavelength_0)][2],
                                                                 grid_velocity_average_profile_line, 0) / sum(g_line)

            average_profile_J_error += g_line[i] * interpol_spec(lines_profile[str(wavelength_0)][2],
                                                                 self.modified_smoothed_line_profiles[
                                                                     str(wavelength_0)][3],
                                                                 grid_velocity_average_profile_line, 0) / sum(g_line)

        return pd.DataFrame(
            {'vel': grid_velocity_average_profile_line, 'V': average_profile_V, 'J': average_profile_J,
             'V_err': average_profile_V_error, 'J_err': average_profile_J_error})

    def differential_method(self, i_stockes, v_stockes, grid_abscissa, type_grid, type_integrate, type_modified):
        if type_modified:
            if type_integrate == 'whole':
                field_factor = i_stockes
                polarization_factor = v_stockes
            else:
                long_magnetic_field_by_mdm_single_line = {}
                for wavelength_0 in self.spectrum_mask['lambda']:
                    tr1, measur_magnetic_field, tr2, err_magnetic_field, tr4 = regress(
                        self.modified_smoothed_line_profiles[str(wavelength_0)][1],
                        self.modified_smoothed_line_profiles[str(wavelength_0)][0])
                    long_magnetic_field_by_mdm_single_line[str(wavelength_0)] = [measur_magnetic_field,
                                                                                 err_magnetic_field]
                return long_magnetic_field_by_mdm_single_line
        else:
            der_i_stockes = interpol_spec(grid_abscissa, i_stockes, grid_abscissa, 1)
            field_factor = (1.0 / i_stockes) * der_i_stockes
            polarization_factor = v_stockes / i_stockes

            if type_grid == 'wavelength':
                if type_integrate == 'whole':
                    g_eff = np.mean(self.spectrum_mask['g_factor'])
                if type_integrate == 'single':
                    g_eff = np.ones(len(grid_abscissa)) * np.mean(self.spectrum_mask['g_factor'])
                    for i in range(len(self.spectrum_mask['lambda'])):
                        wavelength_0 = self.spectrum_mask['lambda'][i]
                        g_factor = self.spectrum_mask['g_factor'][i]
                        left_bound = self.min_velocity_for_analyse * wavelength_0 / speed_of_light + wavelength_0
                        right_bound = self.max_velocity_for_analyse * wavelength_0 / speed_of_light + wavelength_0
                        indx_not_zeros = np.where((grid_abscissa >= left_bound) & (grid_abscissa <= right_bound))
                        g_eff[indx_not_zeros] = g_factor
                field_factor *= K_0 * g_eff * np.power(grid_abscissa, 2.0)
            if type_grid == 'velocity':
                if type_integrate == 'whole':
                    g_eff = np.mean(self.spectrum_mask['g_factor'])
                field_factor *= -R_0 * g_eff * np.power(grid_abscissa, 1.0)

        m = regress(field_factor, polarization_factor)

        if not type_modified:

            if type_grid == 'wavelength':

                if type_integrate == 'whole':

                    fig = plt.figure(figsize=(10, 7))
                    plt.subplots_adjust(hspace=0.45)
                    ax1 = fig.add_subplot(2, 1, 1)
                    ax2 = fig.add_subplot(2, 1, 2)

                    ax1.set_title(r'$<B_z> =\, $' + str(round(m[1], 0)) + r'$\,\pm\,$' + str(round(m[3], 0)) + r' Гс$,\, \chi^2 =\, $' + str(round(m[4], 1)), fontsize=font_size)
                    ax1.set_xlabel(
                        r'$-4.67\cdot10^{-13} g_{eff} \lambda^2 \frac{1}{I} \frac{\mathrm{d}I}{\mathrm{d}\lambda}$',
                        fontsize=font_size)
                    ax1.set_ylabel(r'$V/I$', fontsize=font_size)
                    ax1.plot(field_factor, polarization_factor, 'bs', ms=0.7)
                    ax1.errorbar(field_factor, polarization_factor, yerr=m[2], ecolor='g', elinewidth=0.7, fmt="none",
                                 markeredgewidth=0)
                    ax1.plot(field_factor, m[1] * field_factor + m[0], 'r', lw=1)

                    ax2.set_xlabel(r'$\lambda,\, \AA$', fontsize=font_size)
                    ax2.set_ylabel(r'$I,\, V/I\cdot5+1.05$', fontsize=font_size)
                    ax2.plot(grid_abscissa, i_stockes, 'r-', label='Параметр Стокса I')
                    ax2.plot(grid_abscissa, v_stockes * 5 + 1.05, 'g', label=r'$V/I\cdot5+1.05$')

                    ax2.legend(fontsize=font_size)

                    sf = ScalarFormatter()
                    sf.set_powerlimits((-1, 1))

                    ax1.yaxis.set_major_formatter(sf)

                    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
                        label.set_fontsize(font_size)

                    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
                        label.set_fontsize(font_size)
                    plt.savefig('./Pictures/' + self.star_name + '_DM' + '.png', format='png', dpi=800)
                    plt.close()

        return m[1], m[3]

    def integral_method(self, i_stockes, v_stockes, grid_abscissa, lines_profiles, type_grid, type_integrate,
                        type_modified):
        if type_modified:
            if type_integrate == 'whole':
                alpha_V = trapezoid(grid_abscissa * v_stockes, grid_abscissa)
                alpha_J = trapezoid(grid_abscissa * i_stockes, grid_abscissa)

                error_magnetic_field = measure_error_long_magnetic_field(grid_abscissa, i_stockes, v_stockes,
                                                                         self.average_line_profile['J_err'],
                                                                         self.average_line_profile['V_err'])

            else:
                long_magnetic_field_by_mim_single_line = {}
                for wavelength_0 in self.spectrum_mask['lambda']:
                    M_V = trapezoid(self.lines_profile[str(wavelength_0)][2] *
                                    self.modified_smoothed_line_profiles[str(wavelength_0)][0],
                                    self.lines_profile[str(wavelength_0)][2])
                    M_J = trapezoid(self.lines_profile[str(wavelength_0)][2] *
                                    self.modified_smoothed_line_profiles[str(wavelength_0)][1],
                                    self.lines_profile[str(wavelength_0)][2])

                    error_magnetic_field, bz = measurements_error_long_magnetic_field_integral_methods(
                        self.modified_smoothed_line_profiles[str(wavelength_0)][1] / np.mean(
                            self.spectrum_mask['g_factor']) / np.mean(self.spectrum_mask['lambda']) / R_0,
                        self.modified_smoothed_line_profiles[str(wavelength_0)][0],
                        self.lines_profile[str(wavelength_0)][2],
                        self.modified_smoothed_line_profiles[str(wavelength_0)][2],
                        self.modified_smoothed_line_profiles[str(wavelength_0)][3],
                        np.mean(self.spectrum_mask['g_factor']), np.mean(self.spectrum_mask['lambda']))

                    long_magnetic_field_by_mim_single_line[str(wavelength_0)] = [M_V / M_J, error_magnetic_field]
                return long_magnetic_field_by_mim_single_line
        else:
            if type_grid == 'velocity':
                if type_integrate == 'single':
                    long_magnetic_field_by_im_single_line = {}
                    for i in range(len(self.spectrum_mask['lambda'])):
                        wavelength_0 = self.spectrum_mask['lambda'][i]
                        g_eff = self.spectrum_mask['g_factor'][i]
                        alpha_v = trapezoid(lines_profiles[str(wavelength_0)][4] * lines_profiles[str(wavelength_0)][2],
                                            lines_profiles[str(wavelength_0)][2])
                        alpha_j = -R_0 * g_eff * wavelength_0 * trapezoid(lines_profiles[str(wavelength_0)][3] - 1.0,
                                                                          lines_profiles[str(wavelength_0)][2])

                        error_magnetic_field, bz = measurements_error_long_magnetic_field_integral_methods(
                            lines_profiles[str(wavelength_0)][3],
                            lines_profiles[str(wavelength_0)][4] / lines_profiles[str(wavelength_0)][3],
                            lines_profiles[str(wavelength_0)][2], lines_profiles[str(wavelength_0)][5],
                            lines_profiles[str(wavelength_0)][6], g_eff, wavelength_0)

                        long_magnetic_field_by_im_single_line[str(wavelength_0)] = [alpha_v / alpha_j,
                                                                                    error_magnetic_field]

                    return long_magnetic_field_by_im_single_line
                if type_integrate == 'whole':
                    g_eff = np.mean(self.spectrum_mask['g_factor'])
                    mean_wavelength = np.mean(self.spectrum_mask['lambda'])

                    # alpha_V = trapezoid(grid_abscissa * v_stockes * i_stockes, grid_abscissa)
                    # alpha_J = trapezoid(i_stockes, grid_abscissa) * R_0 * g_eff * mean_wavelength

                    error_magnetic_field, bz = measurements_error_long_magnetic_field_integral_methods(i_stockes,
                                                                                                       v_stockes,
                                                                                                       grid_abscissa,
                                                                                                       self.least_square_deconvolution_line_profile[
                                                                                                           3],
                                                                                                       self.least_square_deconvolution_line_profile[
                                                                                                           4], g_eff,
                                                                                                       mean_wavelength)

                    alpha_V = bz
                    alpha_J = 1.0

                else:
                    return
            else:
                return

        return [alpha_V / alpha_J, error_magnetic_field]

    def compute_magnetic_field(self, method, smoothed_parameter_k):
        if method == 'IM_single':
            if self.lines_profile is None:
                self.lines_profile = self.compute_line_profile()
            self.long_magnetic_field_by_im_single_line = self.integral_method(None, None, None, self.lines_profile,
                                                                              'velocity', 'single', False)

        if method == 'MIM_whole':
            if self.lines_profile is None:
                self.lines_profile = self.compute_line_profile()
            if self.average_line_profile is None:
                self.average_line_profile = self.compute_average_smoothed_line_profile(smoothed_parameter_k,
                                                                                       self.lines_profile)
            self.long_magnetic_field_by_mim_whole_spectrum = self.integral_method(self.average_line_profile['J'],
                                                                                  self.average_line_profile['V'],
                                                                                  self.average_line_profile['vel'],
                                                                                  None, None, 'whole',
                                                                                  True)

        if method == 'MIM_single':
            if self.lines_profile is None:
                self.lines_profile = self.compute_line_profile()
            if self.modified_smoothed_line_profiles is None:
                self.average_line_profile = self.compute_average_smoothed_line_profile(
                    smoothed_parameter_k,
                    self.lines_profile)
            self.long_magnetic_field_by_mim_single_line = self.integral_method(None, None, None,
                                                                               self.modified_smoothed_line_profiles,
                                                                               None, 'single', True)

        if method == 'LSD_IM':
            if self.least_square_deconvolution_line_profile is None:
                self.least_square_deconvolution_line_profile = self.compute_least_square_deconvolution_line_profile()
            self.long_magnetic_field_by_lsd_im_whole_spectrum = self.integral_method(
                self.least_square_deconvolution_line_profile[1], self.least_square_deconvolution_line_profile[2],
                self.least_square_deconvolution_line_profile[0], None, 'velocity', 'whole', False)

        if method == 'MDM_whole':
            if self.lines_profile is None:
                self.lines_profile = self.compute_line_profile()
            if self.average_line_profile is None:
                self.average_line_profile = self.compute_average_smoothed_line_profile(smoothed_parameter_k,
                                                                                       self.lines_profile)
            self.long_magnetic_field_by_mdm_whole_spectrum = self.differential_method(
                self.average_line_profile['J'],
                self.average_line_profile['V'],
                None, None, 'whole', True)

        if method == 'DM_whole':
            self.long_magnetic_field_by_dm_whole_spectrum = self.differential_method(self.i_stockes, self.v_stockes,
                                                                                     self.grid_wavelength,
                                                                                     'wavelength',
                                                                                     'whole', False)

        if method == 'MDM_single':
            if self.lines_profile is None:
                self.lines_profile = self.compute_line_profile()
            if self.average_line_profile is None:
                self.average_line_profile = self.compute_average_smoothed_line_profile(smoothed_parameter_k,
                                                                                       self.lines_profile)

            self.long_magnetic_field_by_mdm_single_line = self.differential_method(None, None, None, None, 'single',
                                                                                   True)

    def compute_magnetic_field_by_integral_method(self):
        if self.lines_profile is None:
            self.lines_profile = self.compute_line_profile()
        self.long_magnetic_field_by_im_single_line = self.integral_method(None, None, None, self.lines_profile,
                                                                          'velocity', 'single', False)

        result_measur = np.array(list(self.long_magnetic_field_by_im_single_line.values()))

        long_magnetic_field = np.mean(result_measur[:, 0])
        error_long_magnetic_field = np.sqrt(np.var(result_measur[:, 0]))

        return long_magnetic_field, error_long_magnetic_field

    def compute_magnetic_field_by_method(self, method):
        smooth_parameter_k = 0

        if method != 'IM_single':
            self.compute_magnetic_field(method, smooth_parameter_k)

        if method == 'MIM_whole':
            result_magnetic_field, result_error_magnetic_field = self.long_magnetic_field_by_mim_whole_spectrum[0], \
                self.long_magnetic_field_by_mim_whole_spectrum[1]
        elif method == 'MDM_whole':
            result_magnetic_field, result_error_magnetic_field = self.long_magnetic_field_by_mdm_whole_spectrum[0], \
                self.long_magnetic_field_by_mdm_whole_spectrum[1]
        elif method == 'LSD_IM':
            result_magnetic_field, result_error_magnetic_field = self.long_magnetic_field_by_lsd_im_whole_spectrum[0], \
                self.long_magnetic_field_by_lsd_im_whole_spectrum[1]
        elif method == 'DM_whole':
            result_magnetic_field, result_error_magnetic_field = self.long_magnetic_field_by_dm_whole_spectrum[0], \
                self.long_magnetic_field_by_dm_whole_spectrum[1]
        elif method == 'IM_single':
            result_magnetic_field, result_error_magnetic_field = self.compute_magnetic_field_by_integral_method()
        else:
            print('Wrong method this does not existed or realized in this method')
            return

        return round(result_magnetic_field, 1), round(result_error_magnetic_field, 0)


def DER_SNR(flux):
    # =====================================================================================
    """
    DESCRIPTION This function computes the signal to noise ratio DER_SNR following the
                definition set forth by the Spectral Container Working Group of ST-ECF,
            MAST and CADC.

                signal = median(flux)
                noise  = 1.482602 / sqrt(6) median(abs(2 flux_i - flux_i-2 - flux_i+2))
            snr    = signal / noise
                values with padded zeros are skipped

    USAGE       snr = DER_SNR(flux)
    PARAMETERS  none
    INPUT       flux (the computation is unit independent)
    OUTPUT      the estimated signal-to-noise ratio [dimensionless]
    USES        numpy
    NOTES       The DER_SNR algorithm is an unbiased estimator describing the spectrum
            as a whole as long as
                * the noise is uncorrelated in wavelength bins spaced two pixels apart
                * the noise is Normal distributed
                * for large wavelength regions, the signal over the scale of 5 or
              more pixels can be approximated by a straight line

                For most spectra, these conditions are met.

    REFERENCES  * ST-ECF Newsletter, Issue #42:
                www.spacetelescope.org/about/further_information/newsletters/html/newsletter_42.html
                * Software:
            www.stecf.org/software/ASTROsoft/DER_SNR/
    AUTHOR      Felix Stoehr, ST-ECF
                24.05.2007, fst, initial import
                01.01.2007, fst, added more help text
                28.04.2010, fst, return value is a float now instead of a numpy.float64
    """
    from numpy import array, where, median, abs

    flux = array(flux)

    # Values that are exactly zero (padded) are skipped
    flux = array(flux[where(flux != 0.0)])
    n = len(flux)

    # For spectra shorter than this, no value can be returned
    if n > 4:
        signal = median(flux)

        noise = 0.6052697 * median(abs(2.0 * flux[2:n - 2] - flux[0:n - 4] - flux[4:n]))

        return float(signal / noise)

    else:

        return 0.0


def interpol_spec(w, r, wi, d):
    tck = interpolate.splrep(w, r, s=0)
    ri = interpolate.splev(wi, tck, der=d)
    return ri


def regress(x, y):
    xsq = x ** 2
    N = len(x)
    det = N * sum(xsq) - (sum(x)) ** 2
    a = ((sum(xsq) * sum(y)) - (sum(x) * sum(x * y))) / det
    b = ((N * sum(x * y)) - (sum(x) * sum(y))) / det
    sigma_y = np.sqrt(sum((y - a - b * x) ** 2) / (N - 2))
    sigma_b = np.sqrt(N * sigma_y ** 2 / det)
    chisq = sum(((y - a - b * x) / sigma_y) ** 2)

    m = np.array([a, b, sigma_y, sigma_b, chisq])
    return m


def core_of_convolution(k):
    core_of_function = lambda x: np.exp(-np.power(x, 2.0) / 2.0)
    function_for_sum = []
    n = k // 2
    coefficient = [
        np.power(-2.0, -j) / float(factorial(j)) / float(factorial(-2 * j + k)) * float(factorial(k)) * np.power(
            -1.0,
            k) for
        j in range(n + 1)]
    function_for_sum = lambda x: sum([np.power(x, k - 2.0 * j) * coefficient[j] for j in range(n + 1)])

    g_k = lambda x: core_of_function(x) * function_for_sum(x) * np.power(-1, k) / np.sqrt(2.0 * np.pi)

    return g_k


def operator_convolution(k, ordinate_for_convolution, abscissa_for_convolution):
    g_k = core_of_convolution(k)
    convolution_function = lambda v, S: trapezoid(
        g_k((v - abscissa_for_convolution) / S) * ordinate_for_convolution,
        abscissa_for_convolution)

    return convolution_function


def read_mask_spectrum(infile, wavelengths_observation, intensity):
    with open(infile, 'r') as f:
        line = f.readline()
        nlines = int(line.split(',')[2])
        wl, gl, depth = np.loadtxt(itertools.islice(f, 2, nlines + 2), delimiter=',', usecols=(1, 8, 9),
                                   dtype=float, skiprows=0, unpack=True)
        filter_gl = np.where(gl < 99.)
        wavelengths_filter = wl[filter_gl]
        g_factor_lande_filter = gl[filter_gl]
        depth_filter = depth[filter_gl]
        filter_depth = np.where(depth_filter > 0.0)
        wavelengths_filter = wavelengths_filter[filter_depth]
        g_factor_lande_filter = g_factor_lande_filter[filter_depth]
        depth_filter = depth_filter[filter_depth]
        nlines = len(wavelengths_filter)
        weight = np.zeros((2, nlines), dtype=np.double)
    index_obs = \
        np.where(
            (wavelengths_filter >= wavelengths_observation[0]) & (
                    wavelengths_filter <= wavelengths_observation[-1]))[
            0]

    intensity_norm = np.mean(depth_filter[index_obs])
    g_factor_lande_norm = np.mean(g_factor_lande_filter[index_obs])
    wavelengths_norm = np.mean(wavelengths_filter[index_obs])
    weight[0, :] = (depth_filter * wavelengths_filter) / (intensity_norm * wavelengths_norm)
    weight[1, :] = (depth_filter * g_factor_lande_filter * wavelengths_filter) / (
            intensity_norm * wavelengths_norm * g_factor_lande_norm)
    return wavelengths_filter, weight, wavelengths_norm, intensity_norm, g_factor_lande_norm


def fill_M_sparse(wavelengths_observation, wavelengths_mask, weight_mask, radial_velocity):
    light_speed = 299792.4580
    num_spectrum_point = len(wavelengths_observation)
    num_radial_velocity_point = len(radial_velocity)
    index_observation = np.where(
        (wavelengths_mask > wavelengths_observation[0]) & (wavelengths_mask < wavelengths_observation[-1]))
    wavelengths_mask = wavelengths_mask[index_observation[0]]
    weight_mask = weight_mask[index_observation[0]]
    _, index = np.unique(wavelengths_mask, return_index=True)
    wavelengths_mask = wavelengths_mask[index]
    weight_mask = weight_mask[index]
    M = np.zeros((num_spectrum_point, num_radial_velocity_point), dtype=np.double)
    for counter in range(len(wavelengths_mask)):
        velocity_i = light_speed * (wavelengths_observation - wavelengths_mask[counter]) / wavelengths_mask[counter]
        for j in range(num_radial_velocity_point - 1):
            sub_range = np.where((velocity_i >= radial_velocity[j]) & (velocity_i <= radial_velocity[j + 1]))
            for i in sub_range:
                difference = (radial_velocity[j + 1] - radial_velocity[j])
                M[i, j] = weight_mask[counter] * (radial_velocity[j + 1] - velocity_i[i]) / difference
                M[i, j + 1] = weight_mask[counter] * (velocity_i[i] - radial_velocity[j]) / difference
    return sp.csr_matrix(M)


def fill_S_sparse(wl_obs, sigma):
    S2 = sp.diags((1. / sigma) ** 2., dtype=np.double)
    return S2


def solve_least_square_deconvolution_sparse(Y0, S2, M, type, alpha):
    R = sp.csr_matrix(sp.diags([2, -1, -1], [0, 1, -1], shape=(M.shape[1], M.shape[1])))
    R[0, 0] = 1.0
    R[-1, -1] = 1.0

    if (type == 'i'):
        Y0 = np.matrix((1.0 - Y0)).T
    else:
        Y0 = np.matrix(Y0).T
    prod = (M.T @ S2 @ M) + R.multiply(alpha)
    Ainv = np.linalg.pinv(prod.todense())
    Z = Ainv @ (M.T @ S2 @ Y0)
    Yr = M @ Z
    diff = Y0 - Yr
    chisq = (diff.transpose() * S2 * diff).item() / len(Y0)
    if np.sqrt(chisq) > 1:
        lsd_errb = np.sqrt(np.diag(Ainv)) * np.sqrt(chisq)
    else:
        lsd_errb = np.sqrt(np.diag(Ainv))
    return Z, chisq, lsd_errb


def renormalize_iraf(w_init, r_init, fit_func, fit_ord, fit_niter, fit_low_rej, fit_high_rej, window):
    r_0 = r_init.copy()
    r_init = filter_result(r_0, window)
    cont_lev = np.zeros(len(r_init))
    if fit_niter <= 1:
        fit_niter = fit_niter + 1
    w_tmp = w_init
    r_tmp = r_init
    for j in range(fit_niter - 1):
        coef = fit_poly(w_tmp, r_tmp, fit_func, fit_ord)
        cont = fit_cont(w_tmp, fit_func, coef)
        w_tmp, r_tmp, cont, coef = reject_points(w_tmp, r_tmp, cont, fit_low_rej, fit_high_rej, fit_func, fit_ord)
        cont_full = fit_cont(w_init, fit_func, coef)
    cont_lev = cont_full
    idx_wrong = np.where(cont_lev == 0.)[0]
    cont_lev[idx_wrong] = r_init[idx_wrong]
    return 1. - r_0 / cont_lev


def fit_poly(w, r, type, order):
    if type == "legendre":
        coef = legfit(w, r, order)
        return coef
    elif type == "chebyshev":
        coef = chebfit(w, r, order)
        return coef


def reject_points(w, r, cont, low_rej, high_rej, func, ord):
    resid = r - cont
    stdr = np.std(resid)
    idx = np.where((resid >= -low_rej * stdr) & (resid <= high_rej * stdr))
    coef = fit_poly(w[idx], r[idx], func, ord)
    return w[idx], r[idx], cont, coef


def filter_result(input_signal, window):
    """ Function smooths input data using Savitsky-Golay filter
    """
    return signal.savgol_filter(input_signal, window_length=window, polyorder=2, axis=0, mode='nearest')


def fit_cont(w, type, coef):
    if type == "legendre":
        return legval(w, coef)
    elif type == "chebyshev":
        return chebval(w, coef)


def measure_error_long_magnetic_field(vel, Zi, Zv, Zi_err, Zv_err):
    uncZi = unumpy.uarray(Zi, Zi_err)
    uncZv = unumpy.uarray(Zv, Zv_err)
    intV = integrate.trapezoid(y=vel * uncZv)
    intI = integrate.trapezoid(y=vel * uncZi)
    uncBz = intV / intI
    return uncBz.s


if __name__ == '__main__':
    # Считывание маски спектра

    Spectrum_mask_array = ['HD_109995_mask.csv', 'HD_109995_mask.csv', 'BD+422309_mask.csv', 'BD+302431_mask.csv',
                           'HD_167105_mask.csv', 'BD+252602_mask.csv', 'HD_128801_mask.csv']

    radial_velocity_array = [-129.27, -129.27, -145.96, -27.46, -173.56, -71.63, -88.04]
    vsini_array = [26.43, 26.43, 32.40, 20.03, 20.03, 26.74, 17.92]

    star_name_array = ['HD_109995', 'rHD_109995', 'BD+422309', 'BD+302431', 'HD_167105', 'BD+252602', 'HD_128801']

    file_name_fits_array = ['HD109995', 'rHD109995', 'BD422309', 'BD302431', 'HD167105', 'BD252602', 'HD128801']

    for i in range(len(star_name_array)):

        Spectrum_mask = Spectrum_mask_array[i]
        file_name_fits = file_name_fits_array[i]
        radial_velocity = radial_velocity_array[i]
        vsini = vsini_array[i]
        star_name = star_name_array[i]

        line_parameter = pd.read_csv(Spectrum_mask, sep='|')

        wavelengths, intensity_pol_l = read_spectrum_sao('./Spectrum/' + file_name_fits + '_1.fits')

        _, intensity_pol_r = read_spectrum_sao('./Spectrum/' + file_name_fits + '_2.fits')

        wavelengths_corr = wavelengths / (1.0 + radial_velocity / speed_of_light)

        star = polar_spectrum_star(wavelengths_corr, intensity_pol_l, intensity_pol_r, line_parameter, -90.0, 70.0,
                                   vsini, star_name)

        print(star_name)

        print('Diff Method', star.compute_magnetic_field_by_method('DM_whole'))
        print('Mod Diff Method', star.compute_magnetic_field_by_method('MDM_whole'))
        print('Mod Int Method', star.compute_magnetic_field_by_method('MIM_whole'))
        print('LSD method', star.compute_magnetic_field_by_method('LSD_IM'))

        # for wavelength_0 in star.spectrum_mask['lambda']:
        #     plt.plot(star.lines_profile[str(wavelength_0)][2], star.lines_profile[str(wavelength_0)][3], label=str(wavelength_0))
        #     plt.legend()
        #     plt.show()

        fig = plt.figure(figsize=(11, 7))
        plt.subplots_adjust(hspace=0.45)

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.grid()
        ax2.grid()

        sf = ScalarFormatter()
        sf.set_powerlimits((-1, 1))

        ax1.yaxis.set_major_formatter(sf)
        ax2.yaxis.set_major_formatter(sf)

        ax1.plot(star.average_line_profile['vel'], star.average_line_profile['V'], color='black')
        ax1.set_xlabel('v, км/с', fontsize=font_size)
        ax1.set_ylabel(r'V$_S$(v, S)', fontsize=font_size)

        ax2.plot(star.average_line_profile['J'], star.average_line_profile['V'], color='black')
        ax2.set_xlabel('J$_S$(v, S)', fontsize=font_size)
        ax2.set_ylabel('V$_S$(v, S)', fontsize=font_size)

        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontsize(font_size)

        for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
            label.set_fontsize(font_size)

        plt.savefig('./Pictures/' + star.star_name + '_MDM' + '.png', format='png', dpi=800)
        plt.close()

        fig = plt.figure(figsize=(9, 7))
        plt.subplots_adjust()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.grid()
        ax2.grid()

        ax1.plot(star.least_square_deconvolution_line_profile[0], star.least_square_deconvolution_line_profile[2],
                 color='black')
        ax1.plot(star.least_square_deconvolution_line_profile[0],
                 np.zeros(len(star.least_square_deconvolution_line_profile[0])), linestyle='--', color='red')
        ax1.set_ylabel('V/I', fontsize=font_size)
        ax1.xaxis.set_major_formatter(NullFormatter())

        sf = ScalarFormatter()
        sf.set_powerlimits((-3, 3))
        ax1.yaxis.set_major_formatter(sf)

        ax2.plot(star.least_square_deconvolution_line_profile[0],
                 1.0 - star.least_square_deconvolution_line_profile[1], color='black')
        ax2.set_xlabel('v, км/с', fontsize=font_size)
        ax2.set_ylabel('I', fontsize=font_size)

        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontsize(font_size)

        for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
            label.set_fontsize(font_size)

        plt.savefig('./Pictures/' + star.star_name + '_LSD' + '.png', format='png', dpi=800)
        plt.close()
