# partitioning.py

# here you can found methods with the purpose to create partitions for the training

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import xgboost as xgb
import numpy as np
import os



# used by generators to create partitions
def sequence_generator(data, seq_length, overlapping, batch_size):
    step = overlapping if overlapping != 0 else seq_length
    X, y = [], []
    
    for i in range(0, len(data) - seq_length, step):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
        
        if len(X) == batch_size:
            yield np.array(X), np.array(y)
            X, y = [], []
    
    if X:
        yield np.array(X), np.array(y)


# create partitions with generators paying attention to the memory occupancy
def create_partitions_with_generators(df, seq_length, test_size, overlapping, folder_path, batch_size, norm=(-1, 1), save_csv=True):
    # Normalizzazione dei dati
    scaler = MinMaxScaler(feature_range=norm)
    df["scaledData"] = scaler.fit_transform(df[["singleData"]])
    
    # Suddivisione del dataset
    total_samples = len(df) - seq_length
    val_test_samples = int(total_samples * test_size)
    train_samples = total_samples - 2 * val_test_samples
    
    train_gen = sequence_generator(df["scaledData"].values[:train_samples], seq_length, overlapping, batch_size)
    val_gen = sequence_generator(df["scaledData"].values[train_samples:train_samples + val_test_samples], seq_length, overlapping, batch_size)
    test_gen = sequence_generator(df["scaledData"].values[train_samples + val_test_samples:], seq_length, overlapping, batch_size)
    
    # Creazione della cartella per salvare i file
    folder_path = os.path.join('npy', folder_path)
    os.makedirs(folder_path, exist_ok=True)
    
    data_partitions = {
        "train": train_gen,
        "val": val_gen,
        "test": test_gen
    }
    
    for partition in data_partitions:
        X_file = os.path.join(folder_path, f'X_{partition}.csv')
        y_file = os.path.join(folder_path, f'y_{partition}.csv')
        buffer_file = os.path.join(folder_path, f'd{partition}.buffer')

        temp_csv_files = []  # Lista per file temporanei
        batch_idx = 0

        # Iterazione sui batch
        for X_batch, y_batch in tqdm(data_partitions[partition], desc=f"Salvataggio {partition.capitalize()}", unit="batch"):
            temp_X_file = f"{X_file}_temp_{batch_idx}.csv"
            temp_y_file = f"{y_file}_temp_{batch_idx}.csv"

            # Salvataggio temporaneo dei batch
            pd.DataFrame(X_batch).to_csv(temp_X_file, index=False, header=False)
            pd.DataFrame(y_batch).to_csv(temp_y_file, index=False, header=False)

            temp_csv_files.append((temp_X_file, temp_y_file))
            batch_idx += 1

        print(f"Unione dei file temporanei per {partition} in un unico buffer...")

        # Unione dei file temporanei in un unico dataset
        X_data = []
        y_data = []
        
        for temp_X_file, temp_y_file in temp_csv_files:
            X_data.append(pd.read_csv(temp_X_file, header=None).values)
            y_data.append(pd.read_csv(temp_y_file, header=None).values.flatten())

            os.remove(temp_X_file)  # Eliminazione del file temporaneo
            os.remove(temp_y_file)  # Eliminazione del file temporaneo

        # Concatenazione dei batch
        X_data = np.vstack(X_data)
        y_data = np.hstack(y_data)

        # Creazione del DMatrix e salvataggio in .buffer
        dmatrix = xgb.DMatrix(X_data, label=y_data)
        dmatrix.save_binary(buffer_file)

        print(f"✅ File .buffer per {partition} salvato correttamente: {buffer_file}")

        # Se save_csv è True, creiamo i file CSV completi
        if save_csv:
            pd.DataFrame(X_data).to_csv(X_file, index=False, header=[f"feature_{i}" for i in range(seq_length)])
            pd.DataFrame(y_data, columns=["target"]).to_csv(y_file, index=False)

    print(f'I file sono stati salvati in {folder_path}')


# create partitions with generators witout paying attention to the memory occupancy
def create_partitions_with_generators_2(df, seq_length, test_size, overlapping, folder_path, batch_size, norm=(-1, 1), save_csv=True):
    scaler = MinMaxScaler(feature_range=norm)
    df["scaledData"] = scaler.fit_transform(df[["singleData"]])
    
    total_samples = len(df) - seq_length
    val_test_samples = int(total_samples * test_size)
    train_samples = total_samples - 2 * val_test_samples
    
    train_gen = sequence_generator(df["scaledData"].values[:train_samples], seq_length, overlapping, batch_size)
    val_gen = sequence_generator(df["scaledData"].values[train_samples:train_samples + val_test_samples], seq_length, overlapping, batch_size)
    test_gen = sequence_generator(df["scaledData"].values[train_samples + val_test_samples:], seq_length, overlapping, batch_size)
    
    folder_path = os.path.join('npy', folder_path)
    os.makedirs(folder_path, exist_ok=True)
    
    data_partitions = {
        "train": train_gen,
        "val": val_gen,
        "test": test_gen
    }
    
    for partition in data_partitions:
        X_file = os.path.join(folder_path, f'X_{partition}.csv')
        y_file = os.path.join(folder_path, f'y_{partition}.csv')
        buffer_file = os.path.join(folder_path, f'd{partition}.buffer')
        
        if save_csv:
            if not os.path.exists(X_file):
                pd.DataFrame(columns=[f"feature_{i}" for i in range(seq_length)]).to_csv(X_file, index=False)
            if not os.path.exists(y_file):
                pd.DataFrame(columns=["target"]).to_csv(y_file, index=False)
        
        X_data, y_data = [], []
        
        for X_batch, y_batch in tqdm(data_partitions[partition], desc=f"Salvataggio {partition.capitalize()}", unit="batch"):
            if save_csv:
                pd.DataFrame(X_batch).to_csv(X_file, mode='a', header=False, index=False)
                pd.DataFrame(y_batch).to_csv(y_file, mode='a', header=False, index=False)
            
            X_data.append(X_batch)
            y_data.append(y_batch)
        
        X_data = np.vstack(X_data)
        y_data = np.hstack(y_data)
        dmatrix = xgb.DMatrix(X_data, label=y_data)
        dmatrix.save_binary(buffer_file)
    
    print(f'I file sono stati salvati in {folder_path}')



# 


















