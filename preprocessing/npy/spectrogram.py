# spectogram

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def calculate_spectrogram(subsignal, fs=50000, nperseg=None, overlap=0.5, plot=False, show_info=False):
    """
    Calcola e visualizza lo spettrogramma di un segnale.

    Parameters:
    - subsignal (np.ndarray): La time series (array 1D) del segnale.
    - fs (int): Frequenza di campionamento in Hz. Default 50000 Hz.
    - nperseg (int): Numero di campioni per ogni segmento della FFT. Default: 1/10 della lunghezza del segnale.
    - overlap (float): Valore normalizzato tra 0 e 1 che indica la proporzione di sovrapposizione tra segmenti.
                       Se 0 nessuna sovrapposizione, se 0.5 il 50%, ecc.
    - plot (bool): Se True, mostra il grafico dello spettrogramma.
    - show_info (bool): Se True, stampa informazioni su forma dello spettrogramma.
    
    Output:
    - f, t, Sxx: Frequenze, tempi e spettrogramma calcolati.
    """
    
    if nperseg is None:
        nperseg = len(subsignal) // 10

    # Controlla che overlap sia un valore valido
    if not 0 <= overlap < 1:
        raise ValueError("Il parametro 'overlap' deve essere compreso tra 0 (incluso) e 1 (escluso).")

    noverlap = int(nperseg * overlap)

    f, t, Sxx = spectrogram(subsignal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    if show_info:
        print(f"nperseg: {nperseg}, overlap: {overlap} -> noverlap: {noverlap}")
        print("Sxx.shape:", Sxx.shape)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequenza [Hz]')
        plt.xlabel('Tempo [s]')
        plt.title('Spettrogramma del segnale')
        plt.colorbar(label='Intensità')
        plt.tight_layout()
        plt.show()

    return f, t, Sxx
























