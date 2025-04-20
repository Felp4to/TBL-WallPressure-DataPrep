# fourier.py

import numpy as np



# extract fft and concatenate real and imag values
def extract_fft_real_imag(window):
    fft = np.fft.fft(window)
    half = len(window) // 2
    real = np.real(fft[:half])
    imag = np.imag(fft[:half])
    return np.concatenate([real, imag])