import numpy as np
import pandas as pd
from dataclasses import dataclass
from astropy.io import fits
from typing import Callable, Tuple
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from Magnetic_field_measurement import polar_spectrum_star

# =====================
# Константы
# =====================
SPEED_OF_LIGHT = 299792.458  # km/s
K0_ZEEMAN = 4.6686e-13  # постоянная для эффекта Зеемана


# =====================
# Dataclass для спектральной линии
# =====================
@dataclass
class SpectralLine:
    element: str
    wavelength: float
    C: float
    sigma: float
    g_factor: float


# =====================
# Модели
# =====================
def gaussian_line(C: float, sigma: float, wavelength_0: float, wavelengths: np.ndarray) -> np.ndarray:
    """Гауссовская линия."""
    return C * np.exp(-0.5 * ((wavelengths - wavelength_0) / sigma) ** 2)


def continuum(wavelengths: np.ndarray) -> np.ndarray:
    """Плоский континуум (единичный)."""
    return np.ones_like(wavelengths)


def add_noise(flux: np.ndarray, snr: float) -> np.ndarray:
    """Добавляет гауссовский шум в спектр."""
    return flux + np.random.normal(0.0, 1.0 / snr, size=flux.shape)


def gaussian_lines_matrix(wavelengths, centers, sigma, depth):
    """
    Векторизованное вычисление суммы гауссовых линий.
    """
    w = wavelengths[None, :]
    c = centers[:, None]

    profiles = depth[:, None] * np.exp(-0.5 * ((w - c) / sigma[:, None]) ** 2)

    return np.sum(profiles, axis=0)


def rotational_kernel(wavelengths, vsini, epsilon=0.6):
    dl = wavelengths[1] - wavelengths[0]
    lambda0 = np.mean(wavelengths)

    delta_lambda = lambda0 * vsini / SPEED_OF_LIGHT

    n = int(delta_lambda / dl)

    grid = np.arange(-n, n + 1) * dl
    x = grid / delta_lambda

    kernel = np.zeros_like(x)

    inside = np.abs(x) < 1

    kernel[inside] = (
            2 * (1 - epsilon) * np.sqrt(1 - x[inside] ** 2)
            + np.pi * epsilon / 2 * (1 - x[inside] ** 2)
    )

    kernel /= np.sum(kernel)

    return kernel


# =====================
# Класс синтетического спектра
# =====================
class SyntheticSpectrum:
    def __init__(self, lines: list[SpectralLine], snr: float = 100.0):
        self.lines = lines
        self.snr = snr

    def spectrum_without_magnetic(self, wavelengths: np.ndarray, vsini: float) -> np.ndarray:
        centers = np.array([l.wavelength for l in self.lines])
        sigma = np.array([l.sigma for l in self.lines])
        depth = np.array([l.C for l in self.lines])

        flux_lines = gaussian_lines_matrix(wavelengths, centers, sigma, depth)

        flux = continuum(wavelengths) + flux_lines

        kernel = rotational_kernel(wavelengths, vsini)

        flux = fftconvolve(flux, kernel, mode="same")

        return add_noise(flux, self.snr)

    def spectrum_with_magnetic(
            self,
            wavelengths: np.ndarray,
            B_field: float,
            vsini: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        # ---- извлекаем параметры линий в массивы ----
        lambda0 = np.array([l.wavelength for l in self.lines])
        depth = np.array([l.C for l in self.lines])
        sigma = np.array([l.sigma for l in self.lines])
        g = np.array([l.g_factor for l in self.lines])

        # ---- Zeeman splitting ----
        delta = K0_ZEEMAN * lambda0 * lambda0 * g * B_field

        centers_L = lambda0 - delta
        centers_R = lambda0 + delta

        # ---- broadcasting для матрицы профилей ----
        w = wavelengths[None, :]  # shape (1, Nλ)

        cL = centers_L[:, None]  # shape (Nline,1)
        cR = centers_R[:, None]

        s = sigma[:, None]
        d = depth[:, None]

        profiles_L = d * np.exp(-0.5 * ((w - cL) / s) ** 2)
        profiles_R = d * np.exp(-0.5 * ((w - cR) / s) ** 2)

        flux_L = np.sum(profiles_L, axis=0)
        flux_R = np.sum(profiles_R, axis=0)

        # ---- continuum ----
        cont = continuum(wavelengths)
        flux_L += cont
        flux_R += cont

        # ---- rotation ----
        kernel_rot = rotational_kernel(wavelengths, vsini)

        flux_L = fftconvolve(flux_L, kernel_rot, mode="same")
        flux_R = fftconvolve(flux_R, kernel_rot, mode="same")

        # ---- noise ----
        flux_L = add_noise(flux_L, self.snr)
        flux_R = add_noise(flux_R, self.snr)

        return flux_L, flux_R

def degrade_resolution(lambda_orig, flux_orig, R, new_delta_lambda=None):
    """
    Понижает разрешение спектра с учётом доплеровского смещения от скорости.

    Параметры:
    ----------
    lambda_orig : array
        Длины волн исходного спектра (Å или нм).
    flux_orig : array
        Поток исходного спектра.
    R : float
        Целевая разрешающая способность (R = λ/Δλ).
    new_delta_lambda : float, optional
        Новый шаг по длине волны (Å/пиксель). Если None, выбирается автоматически.

    Возвращает:
    -----------
    lambda_new : array
        Новая сетка длин волн (с учётом скорости).
    flux_new : array
        Спектр с пониженным разрешением.
    """

    # 2. Размытие спектра
    delta_lambda = np.mean(np.diff(lambda_orig))
    delta_lambda_local = lambda_orig / R
    sigma_lambda = delta_lambda_local / (2 * np.sqrt(2 * np.log(2)))
    sigma_pixels = sigma_lambda / delta_lambda
    flux_smoothed = gaussian_filter1d(flux_orig, np.mean(sigma_pixels))

    # 3. Передискретизация
    if new_delta_lambda is None:
        new_delta_lambda = np.mean(lambda_orig) / R

    lambda_new = np.arange(lambda_orig[0], lambda_orig[-1], new_delta_lambda)
    interp_func = interp1d(lambda_orig, flux_smoothed, kind='cubic', bounds_error=False, fill_value='extrapolate')
    flux_new = interp_func(lambda_new)

    return lambda_new, flux_new


# =====================
# Чтение данных
# =====================
def read_fits_spectrum(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Чтение спектра из FITS."""
    with fits.open(filename) as hdul:
        header = hdul[0].header
        data = hdul[0].data

        n_pixels = header['NAXIS1']
        crval = header['CRVAL1']
        crpix = header['CRPIX1']
        cdelt = header['CDELT1']

        wavelengths = crval + (np.arange(n_pixels) - crpix) * cdelt

        if header['NAXIS'] == 3:
            flux = data[0, 0, :]
        elif header['NAXIS'] == 1:
            flux = data
        else:
            flux = data[0]

    return wavelengths, flux


def read_vald_mask(filename: str, obs_wavelengths: np.ndarray, min_depth: float = 0.1) -> list[SpectralLine]:
    """Чтение маски VALD3 и фильтрация по глубине и Lande-фактору."""
    df = pd.read_csv(filename, delimiter=',', skiprows=3,
                     names=['element', 'lambda', 'gf_unused1', 'gf_unused2', 'unused3', 'unused4',
                            'unused5', 'unused6', 'g_factor', 'depth', 'unused'],
                     usecols=['element', 'lambda', 'g_factor', 'depth'])

    # Фильтр по Lande-фактору и глубине
    df = df[(df.g_factor < 99.0) & (df.depth > min_depth)]

    df['lambda'] = pd.to_numeric(df['lambda'], errors='coerce')
    df['g_factor'] = pd.to_numeric(df['g_factor'], errors='coerce')
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')

    # Фильтр по наблюдаемому диапазону длин волн
    df = df[(df['lambda'] >= obs_wavelengths[0]) & (df['lambda'] <= obs_wavelengths[-1])]

    lines = [SpectralLine(row.element, row['lambda'], -row.depth, 0.01, row.g_factor) for idx, row in df.iterrows()]
    return lines

# =====================
# Пример использования
# =====================
if __name__ == '__main__':
    # # Чтение наблюдаемого спектра

    # wavelengths, flux = read_fits_spectrum('./Spectrum/HD128801_1.fits')

    r = 15000.0

    r_init = 5000000.0
    wavelength_0 = 4650.0
    step_wave = wavelength_0 / r_init

    vel_rot = 30.0
    signal_noise = 1000.0

    wavelengths = np.arange(4400.5, 4899.5, 0.001)

    # Чтение маски VALD3
    lines = read_vald_mask('star_t9000_g4_2l4400_4900.lin', wavelengths)

    # Создание синтетического спектра
    synth = SyntheticSpectrum(lines, snr=signal_noise)

    # Спектр без магнитного поля
    flux_noB = synth.spectrum_without_magnetic(wavelengths, vel_rot)

    flux_L, flux_R = synth.spectrum_with_magnetic(wavelengths, B_field=3e4, vsini=vel_rot)

    wavelengths_res, flux_L = degrade_resolution(wavelengths, flux_L, r)
    _, flux_R = degrade_resolution(wavelengths, flux_R, r)
    _, flux_noB = degrade_resolution(wavelengths, flux_noB, r)

    # Визуализация спектров
    plt.plot(wavelengths_res, flux_noB, label='noB')
    plt.plot(wavelengths_res, flux_L, label='I', linestyle='dashed')
    plt.plot(wavelengths_res, flux_R, label='V', linestyle='dashed')
    plt.legend()
    plt.show()

    # # Пример: сохранить маску в CSV
    # pd.DataFrame([vars(line) for line in lines]).rename(columns={'wavelength': 'lambda','C': 'depth'}).to_csv('test_stars_mask.csv', index=False)
    #
    # Spectrum_mask = 'test_stars_mask.csv'
    # vsini = vel_rot
    # star_name = 'test_star'
    #
    # line_parameter = pd.read_csv(Spectrum_mask, sep=',')
    #
    # star = polar_spectrum_star(wavelengths, flux_L, flux_R, line_parameter, -100.0, 100.0, vsini, star_name)
    #
    # print(star_name)
    #
    # print('Diff Method', star.compute_magnetic_field_by_method('DM_whole'))
    # print('Mod Diff Method', star.compute_magnetic_field_by_method('MDM_whole'))
    # print('Mod Int Method', star.compute_magnetic_field_by_method('MIM_whole'))
    # print('LSD method', star.compute_magnetic_field_by_method('LSD_IM'))
