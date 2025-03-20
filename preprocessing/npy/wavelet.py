# wavelet.py

from tqdm import tqdm
import numpy as np
import pywt



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