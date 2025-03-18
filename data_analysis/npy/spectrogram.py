# spectrogram.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram



# compute spectogram
def compute_spectogram(time_series, nperseg, noverlap, plot):

    # Calcoliamo lo spettrogramma
    f, t, Sxx = spectrogram(time_series, nperseg=nperseg, noverlap=noverlap, scaling='density', mode='magnitude')

    if plot:
        plot_spectogram(f, t, Sxx)

    return f, t, Sxx

# compute spectogram for each window
def compute_spectrogram_windows(time_series, window_size, nperseg, noverlap=50):
    num_windows = len(time_series) // window_size
    spectrogram_windows = []

    for i in range(num_windows):
        window = time_series[i * window_size : (i + 1) * window_size]
        
        # Calcoliamo lo spettrogramma
        f, t, Sxx = spectrogram(window, nperseg=nperseg, noverlap=noverlap, scaling='density', mode='magnitude')
        
        # Prendiamo la densità spettrale di potenza come feature
        spectrogram_windows.append(Sxx.flatten())  # Flatten per renderlo un vettore

    return np.array(spectrogram_windows)

# plot spectogram
def plot_spectogram(f, t, Sxx):
    # Visualizza lo spettrogramma
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f / 1e3, 10 * np.log10(Sxx), shading='gouraud')  # Converti Hz → kHz
    plt.ylabel('Frequenza [kHz]')  # Frequenza in kHz per maggiore leggibilità
    plt.xlabel('Tempo [s]')
    plt.colorbar(label='PSD [dB/Hz]')
    plt.title('Spettrogramma della Time Series')
    plt.show()






















