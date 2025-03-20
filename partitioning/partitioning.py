# partitioning.py

# here you can found methods with the purpose to create partitions for the training

from tqdm import tqdm
import pandas as pd
#import xgboost as xgb
import numpy as np
import json
import os
#import sys

# Aggiunge la cartella superiore al path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocessing', "npy")))

#import normalizzation as norm



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
def create_partitions_with_generators(df, config, folder_path):
    
    # Suddivisione del dataset
    total_samples = len(df) - config["seq_length"]
    val_test_samples = int(total_samples * config["test_ratio"])
    train_samples = total_samples - 2 * val_test_samples
    
    train_gen = sequence_generator(df.values[:train_samples], config["seq_length"], config["overlapping"], config["batch_size"])
    val_gen = sequence_generator(df.values[train_samples:train_samples + val_test_samples], config["seq_length"], config["overlapping"], config["batch_size"])
    test_gen = sequence_generator(df.values[train_samples + val_test_samples:], config["seq_length"], config["overlapping"], config["batch_size"])
    
    # Creazione della cartella per salvare i file
    folder_path = os.path.join('npy', folder_path)
    os.makedirs(folder_path, exist_ok=True)
    
    data_partitions = {
        "train": train_gen,
        "val": val_gen,
        "test": test_gen
    }
    
    shapes = {}
    
    for partition in data_partitions:
        X_file = os.path.join(folder_path, f'X_{partition}.csv')
        y_file = os.path.join(folder_path, f'y_{partition}.csv')

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

        print(f"Unione dei file temporanei per {partition} in un unico file CSV...")

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

        shapes[partition] = (X_data.shape, y_data.shape)

        # Salvataggio dei dati in formato CSV
        pd.DataFrame(X_data).to_csv(X_file, index=False, header=[f"feature_{i}" for i in range(config["seq_length"])] if config["csv_format"] else False)
        pd.DataFrame(y_data, columns=["target"]).to_csv(y_file, index=False)
        
        print(f"✅ File CSV per {partition} salvati correttamente: {X_file}, {y_file}")

    # Salvataggio delle shapes in un file di testo
    shapes_file = os.path.join(folder_path, "shapes.txt")
    with open(shapes_file, "w") as f:
        f.write("Shapes finali delle partizioni:\n")
        for partition, (X_shape, y_shape) in shapes.items():
            f.write(f"{partition}: X -> {X_shape}, y -> {y_shape}\n")
    
    print(f"📏 Shapes finali delle partizioni salvate in {shapes_file}")
    
    # Percorso completo del file
    config_path = os.path.join(folder_path, "config.json")

    # Salvataggio in JSON
    with open(config_path, "w") as file:
        json.dump(config, file, indent=4)

    print(f"Configurazione salvata in {config_path}")
    print(f'I file sono stati salvati in {folder_path}')













