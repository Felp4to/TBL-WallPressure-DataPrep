# fourier.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.signal import spectrogram
from scipy.signal import welch

# compute signal FFT 
def compute_fft(signal: np.ndarray, fs: float):
    n = len(signal)
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(n, d=1/fs)

    # Considera solo le frequenze positive
    pos_mask = freq >= 0
    return freq[pos_mask], fft[pos_mask]


# compute spectrogram
def compute_spectrogram(signal: np.ndarray, fs: float, window_size: int = 8192, overlap: int = 4096, window_type: str = 'hann'):
    f, t, Sxx = spectrogram(signal, fs=fs, window=window_type, nperseg=window_size, noverlap=overlap, scaling='density', mode='magnitude')
    return f, t, Sxx


# Computes FFT and identifies dominant frequency peaks.
def compute_dominant_frequencies(signal, sampling_rate, threshold_ratio=0.2, n_peaks=5, verbose=True):
    """
    Computes FFT and identifies dominant frequency peaks.

    Parameters:
    - signal: Time-domain signal (numpy array)
    - sampling_rate: Sampling rate in Hz
    - threshold_ratio: Relative threshold (0 to 1) for peak detection (default: 0.2)
    - n_peaks: Max number of dominant frequencies to report (default: 5)
    - verbose: Whether to print details (default: True)

    Returns:
    - xf_pos: Positive frequency axis
    - yf_pos: Magnitude spectrum
    - peaks: Indices of the detected peaks
    - dominant_freqs: Array of dominant frequencies in Hz
    - main_freq: Most dominant frequency (Hz) or None
    - period: Period corresponding to main frequency (s) or None
    """
    N = len(signal)
    yf = np.abs(fft(signal))
    xf = fftfreq(N, 1 / sampling_rate)

    xf_pos = xf[:N // 2]
    yf_pos = yf[:N // 2]

    peaks, _ = find_peaks(yf_pos, height=np.max(yf_pos) * threshold_ratio)
    dominant_freqs = xf_pos[peaks]

    if len(dominant_freqs) > 0:
        main_freq = dominant_freqs[0]
        period = 1 / main_freq
        if verbose:
            print(f"Main frequency: {main_freq:.2f} Hz")
            print(f"Corresponding period: {period * 1e3:.3f} ms")
    else:
        main_freq = None
        period = None
        if verbose:
            print("No dominant peaks detected.")

    if verbose:
        print("Dominant frequencies (Hz):", dominant_freqs[:n_peaks])

    return xf_pos, yf_pos, peaks, dominant_freqs, main_freq, period


# Computes the power spectrum using Welch's method and converts it to dB.
def compute_power_spectrum_welch(signal, sampling_rate, nperseg=1024, ref_microbar=True):
    """
    Computes the power spectrum using Welch's method and converts it to dB.

    Parameters:
    - signal: Input time-domain signal (numpy array)
    - sampling_rate: Sampling frequency in Hz
    - nperseg: Length of each segment for Welch method
    - ref_microbar: If True, normalize in dB re: 1 µbar; otherwise dB re: 1

    Returns:
    - frequencies: Frequency axis (Hz)
    - power_spectrum_db: Power spectral density in dB
    """
    frequencies, power_spectrum = welch(signal, fs=sampling_rate, nperseg=nperseg)

    if ref_microbar:
        reference = (1e-6)**2  # 1 µbar RMS squared
        label_unit = "dB re: 1 µbar"
    else:
        reference = 1
        label_unit = "dB re: 1"

    power_spectrum_db = 10 * np.log10(power_spectrum / reference)
    return frequencies, power_spectrum_db, label_unit


# plot spectrogram
def plot_spectrogram(f: np.ndarray, t: np.ndarray, Sxx: np.ndarray, log_scale: bool = True):
    if log_scale:
        Sxx = 10 * np.log10(Sxx + 1e-12)
        label = 'Ampiezza [dB]'
    else:
        label = 'Ampiezza'

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.title('Spettrogramma (STFT)')
    plt.ylabel('Frequenza [Hz]')
    plt.xlabel('Tempo [s]')
    plt.colorbar(label=label)
    plt.tight_layout()
    plt.show()


# Plots the magnitude spectrum and highlights dominant frequencies.
def plot_frequency_spectrum(xf_pos, yf_pos, dominant_freqs):
    plt.figure(figsize=(10, 4))
    plt.plot(xf_pos, yf_pos, label='Spectrum')
    plt.plot(dominant_freqs, yf_pos[np.searchsorted(xf_pos, dominant_freqs)], "ro", label='Peaks')
    plt.title("Spectrum with Highlighted Dominant Frequencies")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Plots the power spectrum in decibels.
def plot_power_spectrum_db(frequencies, power_spectrum_db, label_unit="dB"):
    """
    Plots the power spectrum in decibels.

    Parameters:
    - frequencies: Frequency axis (Hz)
    - power_spectrum_db: Power spectral density in dB
    - label_unit: Label for the dB unit (e.g., 'dB re: 1 µbar')
    """
    plt.figure(figsize=(10, 6))
    plt.semilogx(frequencies, power_spectrum_db, label=f"Power Spectrum ({label_unit})")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(f"Power Spectral Density [{label_unit}]")
    plt.title("Pressure Power Spectrum")
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()