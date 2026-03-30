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

def analyze_mode(posterior_local, beta_loc, i_loc):
    d_beta = beta_loc[1] - beta_loc[0]
    d_i = i_loc[1] - i_loc[0]
    d_bp = bp[1] - bp[0]

    # нормировка локальная
    norm = np.sum(posterior_local) * d_beta * d_i * d_bp
    P = posterior_local / norm

    # маргинализация
    P_beta = np.sum(P, axis=(1, 2)) * d_i * d_bp
    P_i = np.sum(P, axis=(0, 2)) * d_beta * d_bp
    P_bp = np.sum(P, axis=(0, 1)) * d_beta * d_i

    # средние
    beta_mean = np.sum(beta_loc * P_beta) * d_beta
    i_mean = np.sum(i_loc * P_i) * d_i
    bp_mean = np.sum(bp * P_bp) * d_bp

    # σ
    beta_std = np.sqrt(np.sum((beta_loc - beta_mean) ** 2 * P_beta) * d_beta)
    i_std = np.sqrt(np.sum((i_loc - i_mean) ** 2 * P_i) * d_i)
    bp_std = np.sqrt(np.sum((bp - bp_mean) ** 2 * P_bp) * d_bp)

    # доверительные интервалы

    beta_ci = credible_interval(beta_loc, P_beta, d_beta)
    i_ci = credible_interval(i_loc, P_i, d_i)
    bp_ci = credible_interval(bp, P_bp, d_bp)

    return beta_mean, beta_std, i_mean, i_std, bp_mean, bp_std, P_bp, beta_ci, i_ci, bp_ci

def extract_local(posterior, center, window=4):
    ib, ii, _ = center

    b_min = max(0, ib - window)
    b_max = min(num_beta, ib + window + 1)

    i_min = max(0, ii - window)
    i_max = min(num_i, ii + window + 1)

    local = posterior[b_min:b_max, i_min:i_max, :]

    beta_loc = beta_vector[b_min:b_max]
    i_loc = i_vector[i_min:i_max]

    return local, beta_loc, i_loc

def credible_interval(grid, P, dx, alpha=0.68):
    cdf = np.cumsum(P) * dx
    low = grid[np.searchsorted(cdf, (1 - alpha) / 2)]
    high = grid[np.searchsorted(cdf, 1 - (1 - alpha) / 2)]
    return low, high

def is_far(a, b, min_dist=5):
    return (abs(a[0] - b[0]) > min_dist) or (abs(a[1] - b[1]) > min_dist)

if __name__ == '__main__':

    # ==================================================================
    # Подсчет карты, выбор сетки по параметрам. (Расчет пока на фортране.)
    # ==================================================================

    python_compute = False

    num_phases = 72
    num_i = 36
    num_beta = 36
    num_bp0 = 250

    bp = np.linspace(0, 1.0E+4, num_bp0)

    i_vector = np.linspace(0, np.pi, num_i)
    beta_vector = np.linspace(0, np.pi, num_beta)
    phi_vector = np.linspace(0, 1.0, num_phases)

    if python_compute:
        df = pd.read_csv('Test_synt_data.csv')

        observe_data = np.array(list(df['<B_l>']))
        observe_err = np.array(list(df['<B_err>']))

        t_0_1 = time.time()

        posterior_map = magnetic_model.posterior_result(observe_data, observe_err, i_vector, bp, phi_vector)

        num_max = np.argmax(posterior_map)

        indices = np.unravel_index(num_max, posterior_map.shape)

        print(beta_vector[indices[0]] * 180.0 / np.pi, i_vector[indices[1]] * 180.0 / np.pi, bp[indices[2]])

        t_0_2 = time.time()

        print(f'Time compute and plotting: {t_0_2 - t_0_1: .2f} c')

    else:
        data = np.loadtxt("./Fortran_code/fortran_maps_output.dat")

        # проверка на всякий случай
        assert data.shape == (num_beta * num_i, num_bp0)

        # преобразуем в 3D массив
        posterior_map = data.reshape((num_beta, num_i, num_bp0))

    # ==================================================================
    # Шаги сеток
    # ==================================================================
    d_beta = beta_vector[1] - beta_vector[0]
    d_i = i_vector[1] - i_vector[0]
    d_bp = bp[1] - bp[0]

    # ==================================================================
    # Нормировка
    # ==================================================================
    norm = np.sum(posterior_map) * d_beta * d_i * d_bp
    posterior = posterior_map / norm

    # ==================================================================
    # Поиск двух мод
    # ==================================================================
    flat_idx = np.argsort(posterior.ravel())[::-1]

    idx1 = np.unravel_index(flat_idx[0], posterior.shape)

    idx2 = None
    for k in flat_idx[1:]:
        candidate = np.unravel_index(k, posterior.shape)
        if is_far(candidate, idx1):
            idx2 = candidate
            break

    print("Mode 1 index:", idx1)
    print("Mode 2 index:", idx2)

    # ==================================================================
    # Анализ двух веток
    # ==================================================================
    posterior_1, beta_1, i_1 = extract_local(posterior, idx1)
    posterior_2, beta_2, i_2 = extract_local(posterior, idx2)

    res1 = analyze_mode(posterior_1, beta_1, i_1)
    res2 = analyze_mode(posterior_2, beta_2, i_2)

    # ==================================================================
    # Вывод
    # ==================================================================
    print("\n===== MODE 1 =====")
    print(f"beta = {res1[0] * 180.0 / np.pi:.4f} ± {res1[1] * 180.0 / np.pi:.4f}")
    print(f"i    = {res1[2] * 180.0 / np.pi:.4f} ± {res1[3] * 180.0 / np.pi:.4f}")
    print(f"B_p  = {res1[4] :.4f} ± {res1[5]:.4f}")
    print("\n===== 68% credible intervals =====")
    print(f"beta: {res1[7]}")
    print(f"i   : {res1[8]}")
    print(f"B_p : {res1[9]}")

    print("\n===== MODE 2 =====")
    print(f"beta = {res2[0] * 180.0 / np.pi:.4f} ± {res2[1] * 180.0 / np.pi:.4f}")
    print(f"i    = {res2[2] * 180.0 / np.pi:.4f} ± {res2[3] * 180.0 / np.pi:.4f}")
    print(f"B_p  = {res2[4]:.4f} ± {res2[5]:.4f}")
    print("\n===== 68% credible intervals =====")
    print(f"beta: {res2[7]}")
    print(f"i   : {res2[8]}")
    print(f"B_p : {res2[9]}")

    # ==================================================================
    # Сравнение по полю
    # ==================================================================
    print("\n===== FIELD CONSISTENCY =====")
    print(f"ΔB_p = {abs(res1[4] - res2[4]):.4f}")

    # ==================================================================
    # МАРГИНАЛИЗАЦИЯ
    # ==================================================================
    P_beta = np.sum(posterior, axis=(1, 2)) * d_i * d_bp
    P_i = np.sum(posterior, axis=(0, 2)) * d_beta * d_bp
    P_bp = np.sum(posterior, axis=(0, 1)) * d_beta * d_i

    # ==================================================================
    # СРЕДНИЕ
    # ==================================================================
    beta_mean = np.sum(beta_vector * P_beta) * d_beta
    i_mean = np.sum(i_vector * P_i) * d_i
    bp_mean = np.sum(bp * P_bp) * d_bp

    # ==================================================================
    # ДИСПЕРСИИ
    # ==================================================================
    beta_var = np.sum((beta_vector - beta_mean) ** 2 * P_beta) * d_beta
    i_var = np.sum((i_vector - i_mean) ** 2 * P_i) * d_i
    bp_var = np.sum((bp - bp_mean) ** 2 * P_bp) * d_bp

    beta_std = np.sqrt(beta_var)
    i_std = np.sqrt(i_var)
    bp_std = np.sqrt(bp_var)

    # ==================================================================
    # MAP ОЦЕНКА
    # ==================================================================
    idx = np.unravel_index(np.argmax(posterior_map), posterior_map.shape)

    beta_map = beta_vector[idx[0]]
    i_map = i_vector[idx[1]]
    bp_map = bp[idx[2]]

    beta_ci = credible_interval(beta_vector, P_beta, d_beta)
    i_ci = credible_interval(i_vector, P_i, d_i)
    bp_ci = credible_interval(bp, P_bp, d_bp)

    # ==================================================================
    # ВЫВОД
    # ==================================================================
    print("==================================================================")
    print("===== POSTERIOR ESTIMATES (One Mode) =====")

    print(f"beta = {beta_mean * 180.0 / np.pi:.4f} ± {beta_std * 180.0 / np.pi:.4f}")
    print(f"i    = {i_mean * 180.0 / np.pi:.4f} ± {i_std * 180.0 / np.pi:.4f}")
    print(f"B_p  = {bp_mean:.4f} ± {bp_std:.4f}")

    print("\n===== 68% credible intervals =====")
    print(f"beta: {beta_ci }")
    print(f"i   : {i_ci}")
    print(f"B_p : {bp_ci}")

    print("\n===== MAP =====")
    print(f"beta = {beta_map * 180.0 / np.pi:.4f}")
    print(f"i    = {i_map * 180.0 / np.pi:.4f}")
    print(f"B_p  = {bp_map:.4f}")

    # ==================================================================
    # 1D маргинальные распределения
    # ==================================================================
    P_beta = np.sum(posterior_map, axis=(1, 2))
    P_i = np.sum(posterior_map, axis=(0, 2))
    P_bp = np.sum(posterior_map, axis=(0, 1))

    # ==================================================================
    # 2D распределения
    # ==================================================================
    P_beta_i = np.sum(posterior_map, axis=2)
    P_beta_bp = np.sum(posterior_map, axis=1)
    P_i_bp = np.sum(posterior_map, axis=0)

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