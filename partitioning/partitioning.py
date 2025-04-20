# partitioning_2.py

import os
import json
import pywt
import sys
import numpy as np

modulo_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'preprocessing', 'npy')
)
sys.path.append(modulo_path)
import spectrogram as spectrogram
import fourier as fourier
import wavelet as wavelet
import time_features as t_features




# 🔀 Shuffle su tutte le partizioni
def shuffle_data(X, y):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], y[idx]

# Split delle partizioni train/val/test SENZA leakage, shuffle SOLO sul training set
def split_and_save_dataset(X, y, foldername, config):
    assert abs(config['train_ratio'] + config['val_ratio'] + config['test_ratio'] - 1.0) < 1e-6, \
        "La somma di train_ratio, val_ratio e test_ratio deve essere 1.0"
    
    # Crea la cartella se non esiste
    os.makedirs(foldername, exist_ok=True)

    # Salva config in un file JSON
    config_path = os.path.join(foldername, "config.txt")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Calcola le dimensioni delle partizioni
    total_samples = len(X)
    train_end = int(total_samples * config['train_ratio'])
    val_end = train_end + int(total_samples * config['val_ratio'])
    
    # Split sequenziale
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test   = X[val_end:], y[val_end:]

    if config.get("shuffle", True):
        X_train, y_train = shuffle_data(X_train, y_train)
        X_val, y_val     = shuffle_data(X_val, y_val)
        X_test, y_test   = shuffle_data(X_test, y_test)

    # Salva le partizioni
    np.savetxt(os.path.join(foldername, "X_train.csv"), X_train, delimiter=",")
    np.savetxt(os.path.join(foldername, "y_train.csv"), y_train, delimiter=",")
    np.savetxt(os.path.join(foldername, "X_val.csv"),   X_val,   delimiter=",")
    np.savetxt(os.path.join(foldername, "y_val.csv"),   y_val,   delimiter=",")
    np.savetxt(os.path.join(foldername, "X_test.csv"),  X_test,  delimiter=",")
    np.savetxt(os.path.join(foldername, "y_test.csv"),  y_test,  delimiter=",")

    # Info partizioni
    info = (
        f"Totale campioni: {total_samples}\n"
        f"Train set: X = {X_train.shape}, y = {y_train.shape}\n"
        f"Validation set: X = {X_val.shape}, y = {y_val.shape}\n"
        f"Test set: X = {X_test.shape}, y = {y_test.shape}\n"
    )

    # Salva info
    with open(os.path.join(foldername, "partitions_info.txt"), 'w') as f:
        f.write(info)

    print(info.strip())
    print(f"\n✅ Partizioni create e salvate in '{foldername}'")


# restituisce una elemento della sequenza in questo formato: {features, fft[window], [t(i), t(i + window_size)]}
def extract_features_stats_raw(window):
    stats = t_features.extract_features(window)
    raw = window
    fft_feats = fourier.extract_fft_real_imag(window)
    return np.concatenate([stats, raw, fft_feats])


# restituisce matrice di sequenze: {features, [t(i), t(i + window_size)], fft[window], wavelet[window]} --> t(i + window_size + 1)
def generate_multi_feature_sequences(ts, ts_normalized, config):  
    X = []
    y = []
    y2 = []

    window_size = config['window_size']
    sequence_length = config['seq_length']
    overlap = config['overlapping']

    base_step = window_size * sequence_length + 1
    assert 0 <= overlap < 1, "overlap deve essere un valore tra 0 e 1 (escluso 1)"
    
    # Calcola lo step effettivo con overlapping
    step = int(base_step * (1 - overlap)) if overlap > 0 else base_step

    max_start = len(ts) - (window_size * sequence_length + 1)
    
    for start in range(0, max_start + 1, step):
        features = []
        valid = True

        for k in range(sequence_length):
            start_index = start + k * window_size
            end_index = start_index + window_size

            if end_index > len(ts_normalized):
                valid = False
                break

            window = ts_normalized[start_index:end_index]
            # estrazione delle feature
            #stats = t_features.extract_time_features(window)
            raw = window
            fft_coeff = fourier.extract_fft_real_imag(window)
            #wavelet_coeff = wavelet.wavelet_transform(window)
            features.append(np.concatenate([raw, fft_coeff]))
            #features.append(np.concatenate([stats, raw, fft_feats, wavelet_coeff]))

        if not valid:
            break

        target_index = start + window_size * sequence_length
        if target_index >= len(ts):
            break

        X.append(np.concatenate(features))
        y.append(ts[target_index])
        y2.append(ts_normalized[target_index])

    X = np.array(X)
    y = np.array(y)
    y2 = np.array(y2)

    print(f"Shape di X: {X.shape}")
    print(f"Shape di y: {y.shape}")

    return X, y, y2


# Ogni sequenza è composta da una serie di valori di pressione, e il target è il valore successivo
def generate_time_sequences(ts, ts_normalized, config):
    seq_length = config["seq_length"]
    overlap = config["overlapping"]
    step = int(seq_length * (1 - overlap) + 1) if overlap > 0 else seq_length + 1

    X, y, y2 = [], [], []

    for i in range(0, len(ts_normalized) - seq_length, step):
        X.append(ts_normalized[i:i + seq_length])
        y.append(ts[i + seq_length])
        y2.append(ts_normalized[i + seq_length])

    X = np.array(X)
    y = np.array(y)
    y2 = np.array(y2)

    # Convert X in 2D (samples x timesteps)
    X = X.reshape(X.shape[0], -1)

    # Convert y in 1D
    y = y.reshape(-1)
    y2 = y2.reshape(-1)
    
    return X, y, y2


# Ogni sequenza è composta da più finestre trasformate con Fourier, e il target è la finestra successiva alla sequenza
def generate_fft_sequences_with_frequency_target(ts, config):
    X = []
    y = []

    window_size = config['window_size']
    sequence_length = config['seq_length']
    overlap = config['overlapping']

    base_step = window_size * sequence_length + window_size
    assert 0 <= overlap < 1, "overlap deve essere un valore tra 0 e 1 (escluso 1)"
    
    # Calcola lo step effettivo con overlapping
    step = int(base_step * (1 - overlap)) if overlap > 0 else base_step

    max_start = len(ts) - (window_size * sequence_length + window_size)

    for start in range(0, max_start + 1, step):
        features = []
        valid = True

        for k in range(sequence_length):
            start_index = start + k * window_size
            end_index = start_index + window_size

            if end_index > len(ts):
                valid = False
                break

            window = ts[start_index:end_index]
            fft_window = np.fft.fft(window)
            fft_features = np.concatenate([np.real(fft_window), np.imag(fft_window)])
            features.append(fft_features)

        if not valid:
            break

        # Target = finestra subito dopo l'ultima finestra di input
        target_start = start + window_size * sequence_length
        target_end = target_start + window_size
        if target_end > len(ts):
            break

        target_window = ts[target_start:target_end]
        fft_target = np.fft.fft(target_window)
        target_features = np.concatenate([np.real(fft_target), np.imag(fft_target)])

        X.append(np.concatenate(features))
        y.append(target_features)

    X = np.array(X)
    y = np.array(y)

    print(f"Shape di X: {X.shape}")
    print(f"Shape di y: {y.shape}")

    return X, y


# Ogni sequenza è composta da più finestre trasformate con Fourier, e il target è il valore temporale successivo alla sequenza.
def generate_fft_sequences_with_temporal_target(ts, config):  
    X = []
    y = []
    
    window_size = config['window_size']
    sequence_length = config['seq_length']
    overlap = config['overlapping']

    base_step = window_size * sequence_length + 1
    assert 0 <= overlap < 1, "overlap deve essere un valore tra 0 e 1 (escluso 1)"

    # Calcola lo step effettivo con overlapping
    step = int(base_step * (1 - overlap)) if overlap > 0 else base_step

    max_start = len(ts) - (window_size * sequence_length + 1)
    
    for start in range(0, max_start + 1, step):
        features = []
        valid = True

        for k in range(sequence_length):
            start_index = start + k * window_size
            end_index = start_index + window_size

            if end_index > len(ts):
                valid = False
                break

            window = ts[start_index:end_index]
            fft_window = np.fft.fft(window)
            fft_features = np.concatenate([np.real(fft_window), np.imag(fft_window)])
            features.append(fft_features)

        if not valid:
            break

        target_index = start + window_size * sequence_length
        if target_index >= len(ts):
            break

        X.append(np.concatenate(features))
        y.append(ts[target_index])

    X = np.array(X)
    y = np.array(y)

    print(f"Shape di X: {X.shape}")
    print(f"Shape di y: {y.shape}")

    return X, y


# Ogni sequenza è composta da più finestre trasformate con Wavelet, e il target è la finestra successiva alla sequenza.
def generate_wavelet_sequences_with_wavelet_target(ts, config):
    window_size = config['window_size']
    overlapping = config['overlapping']
    wavelet = config.get('wavelet', 'db4')
    level = config.get('wavelet_level', 3)
    seq_length = config['seq_length']

    assert 0.0 <= overlapping < 1.0, "overlapping deve essere tra 0 (incluso) e 1 (escluso)"
    step_size = max(1, int(window_size * (1 - overlapping)))

    # Calcolo coefficienti wavelet per tutte le finestre
    wavelet_windows = []
    for i in range(0, len(ts) - window_size + 1, step_size):
        window = ts[i:i + window_size]
        coeffs = pywt.wavedec(window, wavelet=wavelet, level=level)
        features = np.concatenate(coeffs)
        wavelet_windows.append(features)

    wavelet_windows = np.array(wavelet_windows)
    num_windows = wavelet_windows.shape[0]
    num_coeff = wavelet_windows.shape[1]

    print("wavelet_windows: ", len(wavelet_windows))
    print("num_windows: ", num_windows)
    print("num_coeff: ", num_coeff)
    
    X = []
    y = []
    stride = seq_length + 1

    for i in range(0, num_windows - stride + 1, stride):
        X_seq = wavelet_windows[i:i + seq_length].flatten()
        y_seq = wavelet_windows[i + seq_length]
        X.append(X_seq)
        y.append(y_seq)

    return np.array(X), np.array(y)


# Ogni sequenza è composta da più finestre trasformate con Wavelet, e il target è il valore temporale successivo alla sequenza.
def generate_wavelet_sequences_with_temporal_target(ts, config):
    X = []
    y = []

    window_size = config['window_size']
    sequence_length = config['seq_length']
    overlap = config['overlapping']
    wavelet_level = config['wavelet_level']
    wavelet = config['wavelet']

    base_step = window_size * sequence_length + 1  
    if overlap == 0:
        base_step += 1 

    assert 0 <= overlap < 1, "overlap deve essere un valore tra 0 e 1 (escluso 1)"

    step = int(base_step * (1 - overlap)) if overlap > 0 else base_step
    max_start = len(ts) - (window_size * sequence_length + 1)

    for start in range(0, max_start + 1, step):
        features = []
        valid = True

        for k in range(sequence_length):
            start_index = start + k * window_size
            end_index = start_index + window_size
            
            if end_index > len(ts):
                valid = False
                break
            
            window = ts[start_index:end_index]
            
            # Applica la trasformata wavelet discreta
            coeffs = pywt.wavedec(window, wavelet=wavelet, level=wavelet_level)
            wavelet_features = np.concatenate(coeffs)
            features.append(wavelet_features)

        if not valid:
            break

        target_index = start + window_size * sequence_length
        if target_index >= len(ts):
            break

        X.append(np.concatenate(features))
        y.append(ts[target_index])

    X = np.array(X)
    y = np.array(y)

    print(f"Shape di X: {X.shape}")
    print(f"Shape di y: {y.shape}")

    return X, y


# Ogni sequenza è composta da spettrogrammi, e il target è il valore temporale successivo alla sequenza
def generate_spectrogram_sequences_with_temporal_target(ts, config):
    X = []
    y = []

    window_size = config['window_size']
    sequence_length = config['seq_length']
    overlap = config['overlapping']
    overlap2 = config['overlapping_2']
    fs = config['sample_rate']
    nperseg = config['nperseg']

    base_step = window_size * sequence_length + 1
    if overlap == 0:
        base_step += 1

    assert 0 <= overlap < 1, "overlap deve essere un valore tra 0 e 1 (escluso 1)"
    step = int(base_step * (1 - overlap)) if overlap > 0 else base_step
    max_start = len(ts) - (window_size * sequence_length + 1)

    for start in range(0, max_start + 1, step):
        features = []
        valid = True

        for k in range(sequence_length):
            start_index = start + k * window_size
            end_index = start_index + window_size

            if end_index > len(ts):
                valid = False
                break
            
            window = ts[start_index:end_index]

            # Calcola lo spettrogramma
            f, t, Sxx = spectrogram.calculate_spectrogram(window, fs=fs, nperseg=nperseg, overlap=overlap2, plot=False, show_info=False)

            # Appiattisci lo spettrogramma (puoi anche fare medie/statistiche)
            spec_features = Sxx.flatten()
            features.append(spec_features)

        if not valid:
            break

        target_index = start + window_size * sequence_length
        if target_index >= len(ts):
            break

        X.append(np.concatenate(features))
        y.append(ts[target_index])

    X = np.array(X)
    y = np.array(y)

    print(f"Shape di X: {X.shape}")
    print(f"Shape di y: {y.shape}")

    return X, y










