# wavelet.py

from tqdm import tqdm
import numpy as np
import pywt

# calculate coeff by wavelet transform and return numpy vector
def wavelet_transform(signal, wavelet='db4', level=1):
    """
    Applica la trasformata wavelet discreta a un segnale 1D.

    Parameters:
    - signal: array 1D numpy
    - wavelet: tipo di wavelet (default: 'db4')
    - level: livello di decomposizione (default: 3)

    Returns:
    - coeffs: array concatenato dei coefficienti (approssimazione + dettagli)
    """
    # Calcola la DWT
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    
    # Concatenazione dei coefficienti in un unico array
    coeffs_flat = np.concatenate(coeffs)
    
    return coeffs_flat


# calculate CWT on moving windows
def compute_cwt_windows(series, wavelet='morl', scales=np.arange(1, 64), seq_length=50):
    # wavelet: tipi di wavelet
    # scales: array delle scale su cui calcolare CWT
    # seq_length: grandezza della finestra mobile
    X = []
    for i in tqdm(range(len(series) - seq_length), desc="Computing CWT"):
        window = series[i:i+seq_length]  
        coefficients, _ = pywt.cwt(window, scales, wavelet)  
        X.append(coefficients.T)
    return np.array(X)


# concatena parte reale e immaginaria 
def extract_wavelet_real_imag(window, wavelet_name='cmor', scales=None):
    """
    Calcola la CWT (trasformata wavelet continua) e restituisce
    la concatenazione della parte reale e immaginaria dei coefficienti.
    """
    if scales is None:
        scales = np.arange(1, len(window) // 2 + 1) 

    coeffs, _ = pywt.cwt(window, scales=scales, wavelet=wavelet_name)

    real = np.real(coeffs).flatten()
    imag = np.imag(coeffs).flatten()
    
    return np.concatenate([real, imag])