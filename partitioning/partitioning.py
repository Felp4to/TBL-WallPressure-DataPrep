# partitioning.py

# here you can found methods with the purpose to create partitions for the training

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os


def create_sequences(data, seq_length=10, overlapping=True):
    X, y = [], []
    
    if overlapping:
        step = 1  # Shift di 1 elemento (overlapping)
    else:
        step = seq_length  # Shift della lunghezza della sequenza (no overlap)
    
    for i in range(0, len(data) - seq_length, step):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])  # Target è il valore successivo alla sequenza
    
    return np.array(X), np.array(y)


def create_partitions(df, seq_length, test_size, overlapping, folder_path):
    # Normalizza i dati
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["scaledData"] = scaler.fit_transform(df[["singleData"]])

    # Generare i dati di input e target
    X, y = create_sequences(df["scaledData"].values, seq_length, overlapping)

    # Divisione in training e test set (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Reshape per adattare il modello LSTM: (samples, time steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    folder_path = os.path.join('npy', folder_path)
    os.makedirs(folder_path, exist_ok=True)

    # Salvataggio dei dati in file .npy
    np.save(os.path.join(folder_path, 'X_train.npy'), X_train)
    np.save(os.path.join(folder_path, 'X_test.npy'), X_test)
    np.save(os.path.join(folder_path, 'y_train.npy'), y_train)
    np.save(os.path.join(folder_path, 'y_test.npy'), y_test)
    print("I dati sono stati salvati nei file X_train.npy, X_test.npy, y_train.npy e y_test.npy")

    return X_train, X_test, y_train, y_test