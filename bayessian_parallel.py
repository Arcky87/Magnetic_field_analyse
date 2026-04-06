import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time
import os
import glob
import multiprocessing as mp

from magnetic_model import magnetic_model
import Bayesian_approach
from Bayesian_approach import (
    extract_local, analyze_mode, credible_interval,
    get_credible_levels, is_far
)

# ==================================================================
# Параметры сетки (общие для всех файлов)
# ==================================================================
num_phases = 72
num_i = 36
num_beta = 36
num_bp0 = 250

bp = np.linspace(0, 1.0E+4, num_bp0)
i_vector = np.linspace(0, np.pi, num_i)
beta_vector = np.linspace(0, np.pi, num_beta)
phi_vector = np.linspace(0, 1.0, num_phases)

# Шаги сетки для нормировки и интервалов
d_beta = beta_vector[1] - beta_vector[0]
d_i = i_vector[1] - i_vector[0]
d_bp = bp[1] - bp[0]


def process_one_file(filepath):
    """
    Загружает CSV, вычисляет апостериорную карту, находит оценки параметров,
    строит и сохраняет график. Результаты сохраняются в текстовый файл и PDF.
    """
    base = os.path.splitext(os.path.basename(filepath))[0]
    os.makedirs("results", exist_ok=True)

    print(f"Processing: {filepath}")

    # 1. Чтение данных
    df = pd.read_csv(filepath)
    observe_data = np.array(df['Be'])
    observe_err = np.array(df['sigma_Be'])

    # 2. Вычисление апостериорной карты
    t0 = time.time()
    posterior_map = magnetic_model.posterior_result(
        observe_data, observe_err,
        i_vector, beta_vector, bp, phi_vector
    )
    t1 = time.time()
    print(f"  -> Posterior of {base} computed in {t1 - t0:.2f} s")

    # 3. Нормировка
    norm = np.sum(posterior_map) * d_beta * d_i * d_bp
    posterior = posterior_map / norm

    # 4. Поиск двух мод
    flat_idx = np.argsort(posterior.ravel())[::-1]
    idx1 = np.unravel_index(flat_idx[0], posterior.shape)
    idx2 = None
    for k in flat_idx[1:]:
        candidate = np.unravel_index(k, posterior.shape)
        if is_far(candidate, idx1):
            idx2 = candidate
            break

    # 5. Анализ каждой моды
    posterior_1, beta_1, i_1 = extract_local(posterior, idx1)
    res1 = analyze_mode(posterior_1, beta_1, i_1)
    if idx2 is not None:
        posterior_2, beta_2, i_2 = extract_local(posterior, idx2)
        res2 = analyze_mode(posterior_2, beta_2, i_2)
    else:
        res2 = None

    # 6. Маргинальные распределения и статистики
    P_beta = np.sum(posterior, axis=(1, 2)) * d_i * d_bp
    P_i = np.sum(posterior, axis=(0, 2)) * d_beta * d_bp
    P_bp = np.sum(posterior, axis=(0, 1)) * d_beta * d_i

    # Средние
    beta_mean = np.sum(beta_vector * P_beta) * d_beta
    i_mean = np.sum(i_vector * P_i) * d_i
    bp_mean = np.sum(bp * P_bp) * d_bp

    # Дисперсии
    beta_var = np.sum((beta_vector - beta_mean) ** 2 * P_beta) * d_beta
    i_var = np.sum((i_vector - i_mean) ** 2 * P_i) * d_i
    bp_var = np.sum((bp - bp_mean) ** 2 * P_bp) * d_bp
    beta_std = np.sqrt(beta_var)
    i_std = np.sqrt(i_var)
    bp_std = np.sqrt(bp_var)

    # MAP
    idx_map = np.unravel_index(np.argmax(posterior_map), posterior_map.shape)
    beta_map = beta_vector[idx_map[0]]
    i_map = i_vector[idx_map[1]]
    bp_map = bp[idx_map[2]]

    # 68% credible intervals
    beta_ci = credible_interval(beta_vector, P_beta, d_beta)
    i_ci = credible_interval(i_vector, P_i, d_i)
    bp_ci = credible_interval(bp, P_bp, d_bp)

    # 7. Сохранение текстовых результатов
    out_txt = os.path.join("results", base + "_results.txt")
    with open(out_txt, "w",encoding="utf-8") as f:
        f.write(f"File: {filepath}\n")
        f.write("=" * 60 + "\n")
        f.write("MAP estimates\n")
        f.write(f"beta = {beta_map * 180.0 / np.pi:.4f} deg\n")
        f.write(f"i    = {i_map * 180.0 / np.pi:.4f} deg\n")
        f.write(f"B_p  = {bp_map:.4f}\n\n")
        f.write("Posterior mean ± std\n")
        f.write(f"beta = {beta_mean * 180.0 / np.pi:.4f} ± {beta_std * 180.0 / np.pi:.4f} deg\n")
        f.write(f"i    = {i_mean * 180.0 / np.pi:.4f} ± {i_std * 180.0 / np.pi:.4f} deg\n")
        f.write(f"B_p  = {bp_mean:.4f} ± {bp_std:.4f}\n\n")
        f.write("68% credible intervals\n")
        f.write(f"beta: {beta_ci[0] * 180.0 / np.pi:.4f} – {beta_ci[1] * 180.0 / np.pi:.4f} deg\n")
        f.write(f"i   : {i_ci[0] * 180.0 / np.pi:.4f} – {i_ci[1] * 180.0 / np.pi:.4f} deg\n")
        f.write(f"B_p : {bp_ci[0]:.4f} – {bp_ci[1]:.4f}\n\n")
        f.write("Mode 1\n")
        f.write(f"beta = {res1[0] * 180.0 / np.pi:.4f} ± {res1[1] * 180.0 / np.pi:.4f} deg\n")
        f.write(f"i    = {res1[2] * 180.0 / np.pi:.4f} ± {res1[3] * 180.0 / np.pi:.4f} deg\n")
        f.write(f"B_p  = {res1[4]:.4f} ± {res1[5]:.4f}\n")
        f.write("68% intervals: beta: {:.4f}–{:.4f}, i: {:.4f}–{:.4f}, B_p: {:.4f}–{:.4f}\n".format(
            res1[7][0] * 180.0 / np.pi, res1[7][1] * 180.0 / np.pi,
            res1[8][0] * 180.0 / np.pi, res1[8][1] * 180.0 / np.pi,
            res1[9][0], res1[9][1]
        ))
        if res2 is not None:
            f.write("\nMode 2\n")
            f.write(f"beta = {res2[0] * 180.0 / np.pi:.4f} ± {res2[1] * 180.0 / np.pi:.4f} deg\n")
            f.write(f"i    = {res2[2] * 180.0 / np.pi:.4f} ± {res2[3] * 180.0 / np.pi:.4f} deg\n")
            f.write(f"B_p  = {res2[4]:.4f} ± {res2[5]:.4f}\n")
            f.write("68% intervals: beta: {:.4f}–{:.4f}, i: {:.4f}–{:.4f}, B_p: {:.4f}–{:.4f}\n".format(
                res2[7][0] * 180.0 / np.pi, res2[7][1] * 180.0 / np.pi,
                res2[8][0] * 180.0 / np.pi, res2[8][1] * 180.0 / np.pi,
                res2[9][0], res2[9][1]
            ))
            f.write(f"\nΔB_p between modes = {abs(res1[4] - res2[4]):.4f}\n")

    print(f"  -> Saved text results to {out_txt}")

    # 8. Построение и сохранение графика (треугольный plot)
    P_beta_plot = np.sum(posterior_map, axis=(1, 2))
    P_i_plot = np.sum(posterior_map, axis=(0, 2))
    P_bp_plot = np.sum(posterior_map, axis=(0, 1))

    P_beta_i = np.sum(posterior_map, axis=2)
    P_beta_bp = np.sum(posterior_map, axis=1)
    P_i_bp = np.sum(posterior_map, axis=0)

    params = [np.degrees(beta_vector), np.degrees(i_vector), bp]
    labels = [r"$\beta$", r"$i$", r"$B_p$"]
    P_1D = [P_beta_plot, P_i_plot, P_bp_plot]
    P_2D = {
        (1, 0): P_beta_i,
        (2, 0): P_beta_bp,
        (2, 1): P_i_bp
    }

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    # диагональ
    for i in range(3):
        axes[i, i].plot(params[i], P_1D[i], color="black")
        axes[i, i].fill_between(params[i], P_1D[i], color="red", alpha=0.3)
        axes[i, i].set_yticks([])
        axes[i, i].set_title(labels[i], fontsize=12)

    # нижний треугольник
    for (i, j), P in P_2D.items():
        P_smooth = gaussian_filter(P, sigma=1.0)
        lvl_68, lvl_95 = get_credible_levels(P_smooth)
        levels = np.sort(np.unique([lvl_95, lvl_68]))
        axes[i, j].contourf(params[j], params[i], P_smooth.T, levels=30, cmap="Reds")
        if len(levels) >= 2:
            axes[i, j].contour(params[j], params[i], P_smooth.T,
                               levels=levels, colors="black", linewidths=1.2)

    # убрать верхний треугольник
    for i in range(3):
        for j in range(i+1, 3):
            axes[i, j].axis("off")

    # подписи осей
    for j in range(3):
        axes[2, j].set_xlabel(labels[j])
    for i in range(3):
        axes[i, 0].set_ylabel(labels[i])

    # убрать лишние тики
    for i in range(3):
        for j in range(3):
            if i != 2:
                axes[i, j].set_xticklabels([])
            if j != 0:
                axes[i, j].set_yticklabels([])

    plt.tight_layout()
    out_pdf = os.path.join("results", base + "_posterior.pdf")
    plt.savefig(out_pdf)
    plt.close(fig)
    print(f"  -> Saved plot to {out_pdf}")

    return base  # можно вернуть что-то для статистики


if __name__ == "__main__":
    data_folder = "bfield_data"
    file_pattern = os.path.join(data_folder, "*.csv")
    file_list = glob.glob(file_pattern)

    if not file_list:
        print(f"No CSV files found in {data_folder}")
        exit(1)
    
    print(f"Found {len(file_list)} files: {file_list}")

    n_workers = mp.cpu_count()
    print(f"Using {n_workers} parallel workers")

    start_total = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(process_one_file, file_list)

    end_total = time.time()
    print(f"All files processed in {end_total - start_total:.2f} seconds")
