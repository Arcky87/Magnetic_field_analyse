import time

from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from magnetic_model import magnetic_model


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

    # ==================================================================
    # Подсчет карты, выбор сетки по параметрам. (Рассчет пока на фортране.)
    # ==================================================================

    bp = np.linspace(0, 1.0E+4, 250)

    i_vector = np.linspace(0, np.pi, 37)
    beta_vector = np.linspace(0, np.pi, 36)
    phi_vector = np.linspace(0, 2.0 * np.pi, 72)

    observe_data = np.array(list(df['<B_l>']))
    observe_err = np.array(list(df['<B_err>']))

    t_0_1 = time.time()

    prior = magnetic_model.posterior_result(observe_data, observe_err, i_vector, beta_vector, bp, phi_vector)

    num_max = np.argmax(prior)

    indices = np.unravel_index(num_max, prior.shape)

    print(beta_vector[indices[0]] * 180.0 / np.pi, i_vector[indices[1]] * 180.0 / np.pi, bp[indices[2]])

    t_0_2 = time.time()

    print(f'Time compute and plotting: {t_0_2 - t_0_1: .2f} c')

    # ==================================================================
    # 1D маргинальные распределения
    # ==================================================================
    P_beta = np.sum(prior, axis=(1, 2))
    P_i = np.sum(prior, axis=(0, 2))
    P_bp = np.sum(prior, axis=(0, 1))

    # ==================================================================
    # 2D распределения
    # ==================================================================
    P_beta_i = np.sum(prior, axis=2)
    P_beta_bp = np.sum(prior, axis=1)
    P_i_bp = np.sum(prior, axis=0)

    # ==================================================================
    # параметры
    # ==================================================================
    params = [beta_vector, i_vector, bp]
    labels = [r"$\beta$", r"$i$", r"$B_p$"]

    P_1D = [P_beta, P_i, P_bp]

    P_2D = {
        (1, 0): P_beta_i,
        (2, 0): P_beta_bp,
        (2, 1): P_i_bp
    }

    # ==================================================================
    # построение
    # ==================================================================
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