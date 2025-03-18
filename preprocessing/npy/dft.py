# frequency.py

import numpy as np
from tqdm import tqdm
from scipy.signal import spectrogram


# Funzione per generare finestre FFT con ampiezza e fase
def generate_fft_magnitude_phase(time_series, window_size, stride):
    num_windows = (len(time_series) - window_size) // stride + 1
    fft_windows = []

    for i in tqdm(range(num_windows), desc="Computing DFT (Magnitude & Phase)"):
        window = time_series[i * stride : i * stride + window_size]
        fft_result = np.fft.fft(window)                         

        magnitude = np.abs(fft_result)                                  # Ampiezza
        phase = np.angle(fft_result)                                    # Fase

        fft_windows.append(np.concatenate([magnitude, phase]))          # Concateniamo ampiezza + fase

    return np.array(fft_windows)


# Funzione per generare le finestre FFT con parte reale e immaginaria
def generate_fft_re_im(time_series, window_size, stride):
    num_windows = (len(time_series) - window_size) // stride + 1
    fft_windows = []

    for i in tqdm(range(num_windows), desc="Computing DFT (Re e Im)"):
        window = time_series[i * stride : i * stride + window_size]
        fft_result = np.fft.fft(window)                                     # Trasformata di Fourier
        real_part = np.real(fft_result)
        imag_part = np.imag(fft_result)
        fft_windows.append(np.concatenate([real_part, imag_part]))          # Concateniamo real+imag

    return np.array(fft_windows)

# Funzione per calcolare la Trasformata Discreta di Fourier (DFT) su finestre della serie temporale
def compute_dft_windows(time_series, window_size):
    num_windows = len(time_series) // window_size
    dft_windows = []

    for i in tqdm(range(num_windows), desc="Computing DFT"):
        window = time_series[i * window_size : (i + 1) * window_size]
        
        # Calcoliamo la Trasformata Discreta di Fourier (DFT)
        dft_result = np.fft.fft(window)  # FFT per calcolare la DFT
        
        # Prendiamo il modulo dei coefficienti (magnitudine dello spettro)
        dft_magnitude = np.abs(dft_result)

        # Aggiungiamo il risultato alla lista
        dft_windows.append(dft_magnitude)  

    return np.array(dft_windows)

# Funzione per calcolare gli spettrogrammi con una progress bar
def compute_spectrogram_windows(time_series, window_size, nperseg, noverlap=50):
    num_windows = len(time_series) // window_size
    spectrogram_windows = []

    for i in tqdm(range(num_windows), desc="Computing Spectrograms"):
        window = time_series[i * window_size : (i + 1) * window_size]
        
        # Calcoliamo lo spettrogramma
        f, t, Sxx = spectrogram(window, nperseg=nperseg, noverlap=noverlap, scaling='density', mode='magnitude')
        
        # Prendiamo la densità spettrale di potenza come feature
        spectrogram_windows.append(Sxx.flatten())  # Flatten per renderlo un vettore

    return np.array(spectrogram_windows)