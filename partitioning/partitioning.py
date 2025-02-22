# partitioning.py

# here you can found methods with the purpose to create partitions for the training

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
import numpy as np
import os



# Generatore che crea sequenze di dati senza caricarle tutte in memoria.
def sequence_generator(data, seq_length, overlapping, batch_size=1024):
    # Se overlapping è 0, nessuna sovrapposizione tra le sequenze
    step = overlapping if overlapping != 0 else seq_length
    X, y = [], []
    
    # Creazione delle sequenze
    for i in range(0, len(data) - seq_length, step):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])  # La label è il valore successivo alla sequenza
        
        # Quando il batch è pieno, yield e resetta le liste
        if len(X) == batch_size:
            yield np.array(X), np.array(y)
            X, y = [], []
    
    # Se ci sono dati rimanenti, restituisci l'ultimo batch
    if X:
        yield np.array(X), np.array(y)


# Funzione per creare sequenze e salvare i dati in modo ottimizzato.
def create_partitions_with_generators(df, seq_length, test_size, overlapping, folder_path, batch_size):
    """
    Funzione per creare sequenze e salvare i dati in modo ottimizzato.
    Genera solo 4 file finali: X_train, X_test, y_train, y_test
    """

    # Normalizza i dati
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["scaledData"] = scaler.fit_transform(df[["singleData"]])

    # Divide il dataset in train e test
    total_samples = len(df) - seq_length
    train_samples = int(total_samples * (1 - test_size))
    
    train_gen = sequence_generator(df["scaledData"].values[:train_samples], seq_length, overlapping, batch_size)
    test_gen = sequence_generator(df["scaledData"].values[train_samples:], seq_length, overlapping, batch_size)

    folder_path = os.path.join('npy', folder_path)
    os.makedirs(folder_path, exist_ok=True)

    # File CSV per appendere i dati
    train_X_file = os.path.join(folder_path, 'X_train.csv')
    train_y_file = os.path.join(folder_path, 'y_train.csv')
    test_X_file = os.path.join(folder_path, 'X_test.csv')
    test_y_file = os.path.join(folder_path, 'y_test.csv')

    # Salva le intestazioni dei file CSV (se non esistono già)
    if not os.path.exists(train_X_file):
        pd.DataFrame(columns=[f"feature_{i}" for i in range(seq_length)]).to_csv(train_X_file, index=False)
    if not os.path.exists(train_y_file):
        pd.DataFrame(columns=["target"]).to_csv(train_y_file, index=False)
    if not os.path.exists(test_X_file):
        pd.DataFrame(columns=[f"feature_{i}" for i in range(seq_length)]).to_csv(test_X_file, index=False)
    if not os.path.exists(test_y_file):
        pd.DataFrame(columns=["target"]).to_csv(test_y_file, index=False)

    # Scrivi i dati nei file CSV
    for X_batch, y_batch in train_gen:
        # Converte i dati in DataFrame e li appende
        pd.DataFrame(X_batch).to_csv(train_X_file, mode='a', header=False, index=False)
        pd.DataFrame(y_batch).to_csv(train_y_file, mode='a', header=False, index=False)

    for X_batch, y_batch in test_gen:
        pd.DataFrame(X_batch).to_csv(test_X_file, mode='a', header=False, index=False)
        pd.DataFrame(y_batch).to_csv(test_y_file, mode='a', header=False, index=False)

    print(f'I dati sono stati salvati in {folder_path}')



###############
def create_sequences(data, seq_length=10, overlapping=True):
    X, y = [], []
    
    if overlapping:
        step = 1                                                        # Shift di 1 elemento (overlapping)
    else:
        step = seq_length                                               # Shift della lunghezza della sequenza (no overlap)
    
    for i in range(0, len(data) - seq_length, step):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])                                    # Target è il valore successivo alla sequenza
    
    return np.array(X), np.array(y)


def create_partitions(df, seq_length, test_size, overlapping, shuffle, folder_path):
    # Normalizza i dati
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["scaledData"] = scaler.fit_transform(df[["singleData"]])

    # Generare i dati di input e target
    X, y = create_sequences(df["scaledData"].values, seq_length, overlapping)

    # Divisione in training e test set (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Reshape per adattare il modello LSTM: (samples, time steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))  # Rende 2D
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))      # Rende 2D

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    folder_path = os.path.join('npy', folder_path)
    os.makedirs(folder_path, exist_ok=True)

    # Salvataggio dei dati in file .npy
    np.save(os.path.join(folder_path, 'X_train.npy'), X_train)
    np.save(os.path.join(folder_path, 'X_test.npy'), X_test)
    np.save(os.path.join(folder_path, 'y_train.npy'), y_train)
    np.save(os.path.join(folder_path, 'y_test.npy'), y_test)
    dtrain.save_binary(os.path.join(folder_path, 'dtrain.buffer'))
    dtest.save_binary(os.path.join(folder_path, 'dtest.buffer'))

    print(f'I dati sono stati salvati in {folder_path}')

    return X_train, X_test, y_train, y_test






























