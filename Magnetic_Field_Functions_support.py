import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from magnetic_model import magnetic_model

def longitudinal_magnetic_field_landstreet(phase_angles, declination_of_rotation, declination_of_magnetic_field,
                                           polar_magnetic_field, asymptotic_dipole, quad_pole, octo_pole, a_ld=0.35,
                                           b_ld=0.25):
    """
    Программа по рассчету продольного и среднего поля для звезд Ap, написанная J. Landstreet. Программа реализована на фортран, данная оболочка вызывает оригинальную функцию скомпилированную на фортране в питон библиотеку.
    Немного о модификации были поправлены немного вычисления интегралов, сделав их немного точнее. Была увеличена сетка с 8 точек до 40.
    Эффект limb darkening теперь учитывается линейно. Звезда разбивается на секторы равных площадей. На выходе программа выдает вектор, B_l.
    :param phase_angles: numpy.array, вектор фазовых углов, длина может быть произвольной.
    :param declination_of_rotation: float, значения угла наклона оси вращения в радианах.
    :param declination_of_magnetic_field: float, значение угла наклона магнитного диполя относительно оси вращения в радианах.
    :param polar_magnetic_field: float, значение дипольного магнитного поля.
    :param asymptotic_dipole: float, значение асимптотического магнитного поля. Математически ограничение |AD| < 1, более физическое ограничение и ограничение связанное с точностью интегрирования |AD| <= 0.5
    :param quad_pole: float, значение величины квадроупольного магнитного поля. Разумные ограничения |BQ| <= 0.1 - 0.5 B_p.
    :param octo_pole: float, значение величины октоупольного магнитного поля. Разумное ограничение |BOCT| <= 0.3 B_p.
    :param a_ld: float, коэффициенты для линейной интерполяции limb darkening, коэффициент наклона, можно 0, тогда приближение аналогично оригиналу.
    :param b_ld: float, коэффициенты для линейной интерполяции limb darkening, свободный член.
    :return: numpy.array, numpy.array, longitudinal magnetic field.
    """

    coef_to_grad = 180.0 / np.pi

    declination_of_rotation_grad = declination_of_rotation * coef_to_grad
    declination_of_magnetic_field_grad = declination_of_magnetic_field * coef_to_grad

    phases = np.asfortranarray(phase_angles, dtype=float)

    longitudinal_magnetic_field = np.zeros(len(phases), dtype=float)

    declination_of_magnetic_field_grad = float(declination_of_magnetic_field_grad)
    declination_of_rotation_grad = float(declination_of_rotation_grad)
    polar_magnetic_field = float(polar_magnetic_field)
    asymptotic_dipole = float(asymptotic_dipole)
    quad_pole = float(quad_pole)
    octo_pole = float(octo_pole)
    a_ld = float(a_ld)
    b_ld = float(b_ld)

    for i in range(len(phases)):
        longitudinal_magnetic_field[i] = magnetic_model.compute_bl(phases[i], declination_of_rotation_grad,
                                                                         declination_of_magnetic_field_grad,
                                                                         polar_magnetic_field,
                                                                         asymptotic_dipole, quad_pole, octo_pole, a_ld,
                                                                         b_ld)

    return longitudinal_magnetic_field


def c_1(u):
    """
    Вспомогательная функция C_1 из работы arxiv 2404.17517
    :param u: float, numpy.array, Параметр потемнения к краю диска.
    :return: float, numpy.array
    """
    denim = 3.0 - u
    return (15.0 + u) / (20.0 * denim)


def c_2(u):
    """
    Вспомогательная функция C_2 из работы arxiv 2404.17517
    :param u: float, numpy.array, Параметр потемнения к краю диска.
    :return: float, numpy.array
    """
    denim = 3.0 - u
    return 3.0 * (0.77778 - 0.22613 * u) / denim


def c_3(u):
    """
    Вспомогательная функция C_3 из работы arxiv 2404.17517
    :param u: float, numpy.array, Параметр потемнения к краю диска.
    :return: float, numpy.array
    """
    denim = 3.0 - u
    return 3.0 * (0.64775 - 0.23349 * u) / denim


def cos_gamma(declination_of_rotation, declination_of_magnetic_field, phase_angle):
    """
    Вспомогательная функция cos\gamma из работы arxiv 2404.17517
    :param declination_of_rotation: float, угол наклона в радианах оси вращения звезды.
    :param declination_of_magnetic_field: float, угол наклона оси магнитного диполя к оси вращения в радианах.
    :param phase_angle: float, numpy.array фазовы угол.
    :return: float, numpy.array, возращает значение косинуса угла gamma
    """
    return np.cos(declination_of_rotation) * np.cos(declination_of_magnetic_field) + np.sin(
        declination_of_rotation) * np.sin(declination_of_magnetic_field) * np.cos(2.0 * np.pi * phase_angle)


def average_magnetic_field_longitudinal(polar_magnetic_field, declination_of_rotation, declination_of_magnetic_field,
                                        phase_angle, u=0.5):
    """
    :param polar_magnetic_field: float, значение полярного магнитного поля
    :param declination_of_rotation: flat, значение наклона оси в радианах вращения звезды
    :param declination_of_magnetic_field: float, значение наклона в радианах магнитного диполя к оси вращения звезды
    :param phase_angle: float, numpy.array значение фазового угла.
    :param u: float параметр потемнения диска, 0.5
    :return: float, numpy.array массив значений продольного магнитного поля

    Функция написана на основе работы arxiv 2404.17517. Возращает магнитную кривую для дипольного случая магнитного поля.

    """
    return polar_magnetic_field * c_1(u) * cos_gamma(declination_of_rotation, declination_of_magnetic_field,
                                                     phase_angle)


def average_magnetic_field(polar_magnetic_field, declination_of_rotation, declination_of_magnetic_field,
                           phase_angle, u=0.5):
    """
    :param polar_magnetic_field: float, значение полярного магнитного поля
    :param declination_of_rotation: flat, значение наклона оси в радианах вращения звезды
    :param declination_of_magnetic_field: float, значение наклона в радианах магнитного диполя к оси вращения звезды
    :param phase_angle: float, numpy.array значение фазового угла.
    :param u: float параметр потемнения диска, 0.5
    :return: float, numpy.array массив значений среднего магнитного поля на поверхности.

    Функция написана на основе работы arxiv 2404.17517. Возращает магнитную кривую для дипольного случая магнитного поля.

    """

    cos_g_2 = np.power(cos_gamma(declination_of_rotation, declination_of_magnetic_field, phase_angle), 2.0)
    sin_g_2 = 1.0 - cos_g_2

    return polar_magnetic_field * (c_2(u) * cos_g_2 + c_3(u) * sin_g_2)


if __name__ == '__main__':
    u_dark_parameter = 0.5

    num_phase = 1000
    phi_0 = np.random.uniform(0, 1)

    i_rad = np.random.uniform(0.0, np.pi)
    beta_rad = np.random.uniform(0.0, np.pi)

    polar_field = np.random.uniform(1000.0, 8000.0)
    quad_field = 0.0
    octo_field = 0.0
    ad = 0.4

    phases_array = np.sort(np.random.uniform(0, 1, num_phase)) - phi_0

    magnetic_field = average_magnetic_field_longitudinal(polar_field, i_rad, beta_rad, phases_array)

    bl = longitudinal_magnetic_field_landstreet(phases_array, i_rad, beta_rad, polar_field, ad, quad_field, octo_field)

    plt.plot(phases_array, bl, label='Landstreet Magnetic Field')
    plt.plot(phases_array, magnetic_field, label='Original Magnetic Field')
    plt.legend()
    plt.show()
