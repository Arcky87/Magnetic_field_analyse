import time

from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from Magnetic_Field_Functions_support import longitudinal_magnetic_field_landstreet
from magnetic_model import magnetic_model


# =========================================
# Априоры
# =========================================

def prior_polar_magnetic_field(polar_field):
    """
    Априор полярного поля для дипольного случая, на основе Jeffreys prior. Минимальное поле выбирается как 25 Гс.
    Переходное значение. Максимальное значение поля 10^4, что в целом логично.
    :param polar_field: float, numpy.array.
    :return: distribution: float, numpy.array.
    """
    a = 25.0
    polar_magnetic_field_max = 1.0E+4

    return 1.0 / (polar_field + a) / np.log((a + polar_magnetic_field_max) / a)


def prior_declination_of_rotation(declination_of_rotation):
    """
    Обычное классическое распределение для угла наклона вращения
    :param declination_of_rotation: float, numpy.array. In radians
    :return: distribution: float, numpy.array.
    """

    return 0.5 * np.sin(declination_of_rotation)


def prior_declination_of_magnetic_field(declination_of_magnetic_field):
    """
    Классическое равномерное распределение (Почти нет никаких априорных знаний насчет углов). От 0 до 180 градусов.
    (только принимаются значения в радианах, хотя в целом не имеет значения)
    :param declination_of_magnetic_field: float, numpy.array. In radians.
    :return: distribution: float, numpy.array.
    """

    declination_of_magnetic_field_max = np.pi
    declination_of_magnetic_field_min = 0.0

    return 1.0 / (declination_of_magnetic_field_max - declination_of_magnetic_field_min) * np.ones_like(
        declination_of_magnetic_field)


def prior_phase(phase):
    """
    Классическое равномерное распределение. Так как нет никаких оснований на существование выделенной фазы, на которой проходят наблюдения.
    От 0 до 360 градусов. (только программа работает со значениями от 0 до 1, хотя в целом не имеет значения)
    :param phase: float, numpy.array. In radians.
    :return: distribution: float, numpy.array.
    """

    phase_max = 1.0
    phase_min = 0.0

    return 1.0 / (phase_max - phase_min) * np.ones_like(phase)


def prior_scale_coef(b):
    """
    Априорное распределение масштабного множителя для дисперсий ошибок, учесть, что полученные ошибки принадлежат скорее методу.
    Учесть возможное реальное отклонение плюс доп. шумы. Значение от 0.1 до 2.0
    :param b: float, numpy.array.
    :return: distribution: float, numpy.array.
    """

    b_min = 0.1
    b_max = 2.0

    return 1.0 / b / np.log(b_max / b_min)


def likelihood_mod(observe_data, observe_err, model_data, num_observ, b):
    """
    Функция правдоподобия немного модернизированная с учетом масштабного множителя. Если взять его равным 1, получится классическая функция правдоподобия.
    :param observe_data: float, numpy.array observe data.
    :param observe_err: float, numpy.array observe error.
    :param model_data: float, numpy.array model data.
    :param b: float, scale factor.
    :param num_observ: integer, number of observation.
    :return: float, value likelihood.
    """

    return np.power(2.0 * np.pi, -num_observ / 2.0) * np.power(b, num_observ / 2.0) * np.power(np.prod(observe_err),
                                                                                               -1.0) * np.exp(
        -b / 2.0 * np.sum(np.power((observe_data - model_data) / observe_err, 2.0)))


def get_credible_levels(P, levels=[0.68, 0.95]):
    P_flat = P.flatten()
    idx = np.argsort(P_flat)[::-1]
    P_sorted = P_flat[idx]

    cumsum = np.cumsum(P_sorted)
    cumsum /= cumsum[-1]

    values = []
    for lvl in levels:
        values.append(P_sorted[np.searchsorted(cumsum, lvl)])
    return values


if __name__ == '__main__':
    df = pd.read_csv('Test_synt_data.csv')

    bp = np.linspace(0, 1.0E+4, 20)

    i_vector = np.linspace(0, np.pi, 15)
    beta_vector = np.linspace(0, np.pi, 15)
    phi_vector = np.linspace(0, 2.0 * np.pi, 15)

    b_vector = np.linspace(0.1, 2.0, 10)

    observe_data = np.array(list(df['<B_l>']))
    observe_err = np.array(list(df['<B_err>']))

    t_0_1 = time.time()

    prior = magnetic_model.posterior_result(observe_data, observe_err, i_vector, beta_vector, bp, b_vector, phi_vector)

    num_max = np.argmax(prior)

    indices = np.unravel_index(num_max, prior.shape)

    print(beta_vector[indices[0]] * 180.0 / np.pi, i_vector[indices[1]] * 180.0 / np.pi, bp[indices[2]])

    t_0_2 = time.time()

    print(f'Time compute and plotting: {t_0_2 - t_0_1: .2f} c')

    # =========================
    # 1D маргинальные распределения
    # =========================
    P_beta = np.sum(prior, axis=(1, 2))
    P_i = np.sum(prior, axis=(0, 2))
    P_bp = np.sum(prior, axis=(0, 1))

    # =========================
    # 2D распределения
    # =========================
    P_beta_i = np.sum(prior, axis=2)
    P_beta_bp = np.sum(prior, axis=1)
    P_i_bp = np.sum(prior, axis=0)

    # =========================
    # параметры
    # =========================
    params = [beta_vector, i_vector, bp]
    labels = [r"$\beta$", r"$i$", r"$B_p$"]

    P_1D = [P_beta, P_i, P_bp]

    P_2D = {
        (1, 0): P_beta_i,
        (2, 0): P_beta_bp,
        (2, 1): P_i_bp
    }

    # =========================
    # построение
    # =========================
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    # ---- диагональ ----
    for i in range(3):
        axes[i, i].plot(params[i], P_1D[i], color="black")
        axes[i, i].fill_between(params[i], P_1D[i], color="red", alpha=0.3)
        axes[i, i].set_yticks([])
        axes[i, i].set_title(labels[i], fontsize=12)

    # ---- нижний треугольник ----
    for (i, j), P in P_2D.items():

        P_smooth = gaussian_filter(P, sigma=1.0)

        lvl_68, lvl_95 = get_credible_levels(P_smooth)

        levels = np.sort(np.unique([lvl_95, lvl_68]))

        # заливка
        axes[i, j].contourf(
            params[j],
            params[i],
            P_smooth.T,
            levels=30,
            cmap="Reds"
        )

        # контуры
        if len(levels) >= 2:
            axes[i, j].contour(
                params[j],
                params[i],
                P_smooth.T,
                levels=levels,
                colors="black",
                linewidths=1.2
            )

    # ---- убрать верхний треугольник ----
    for i in range(3):
        for j in range(i + 1, 3):
            axes[i, j].axis("off")

    # ---- подписи осей ----
    for j in range(3):
        axes[2, j].set_xlabel(labels[j])

    for i in range(3):
        axes[i, 0].set_ylabel(labels[i])

    # ---- убрать лишние тики ----
    for i in range(3):
        for j in range(3):
            if i != 2:
                axes[i, j].set_xticklabels([])
            if j != 0:
                axes[i, j].set_yticklabels([])

    plt.tight_layout()
    plt.show()