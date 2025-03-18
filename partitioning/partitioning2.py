# partitioning_2.py

import json
import os
import csv
import random
import numpy as np
from sklearn.utils import shuffle


def save_config_to_txt(config, foldername):
    folder_path = os.path.join("./npy", foldername)

    config_path = os.path.join(folder_path, "config.txt")
    
    with open(config_path, "w") as file:
        json.dump(config, file, indent=4)  # Salviamo il dizionario in formato leggibile
    
    print(f"Configurazione salvata in: {config_path}")


def create_train_val_test_sets(fft_data, foldername, config, buffer_size=10000):
    if config['random_seed'] is not None:
        np.random.seed(config['random_seed'])
        random.seed(config['random_seed'])

    total_samples = len(fft_data) - config['seq_length']
    train_end = int(config['train_ratio'] * total_samples)
    val_end = train_end + int(config['val_ratio'] * total_samples)

    folder_path = os.path.join("./npy", foldername)
    os.makedirs(folder_path, exist_ok=True)

    file_paths = {
        "X_train": os.path.join(folder_path, "X_train.csv"),
        "X_val": os.path.join(folder_path, "X_val.csv"),
        "X_test": os.path.join(folder_path, "X_test.csv"),
        "y_train": os.path.join(folder_path, "y_train.csv"),
        "y_val": os.path.join(folder_path, "y_val.csv"),
        "y_test": os.path.join(folder_path, "y_test.csv"),
        "shape_info": os.path.join(folder_path, "dataset_shapes.txt"),  # File per le shape
    }

    file_handlers = {key: open(path, "w", newline="") for key, path in file_paths.items() if key != "shape_info"}
    writers = {key: csv.writer(file) for key, file in file_handlers.items()}

    buffer = []

    # Contatori per la dimensione delle partizioni
    partition_sizes = {
        "train": 0,
        "val": 0,
        "test": 0
    }
    
    # Creiamo le sequenze in memoria temporanea
    for sample_index in range(total_samples):
        X_seq = fft_data[sample_index : sample_index + config['seq_length']].flatten().tolist()
        y_seq = fft_data[sample_index + config['seq_length']].flatten().tolist()

        # Determiniamo a quale set assegnare il campione
        if sample_index < train_end:
            partition = "train"
        elif sample_index < val_end:
            partition = "val"
        else:
            partition = "test"

        # Aggiungiamo al buffer
        buffer.append((X_seq, y_seq, partition))
        
        # Se il buffer è pieno, mischiamo e scriviamo su file
        if len(buffer) >= buffer_size:
            random.shuffle(buffer)
            for X_seq, y_seq, partition in buffer:
                writers[f"X_{partition}"].writerow(X_seq)
                writers[f"y_{partition}"].writerow(y_seq)
                partition_sizes[partition] += 1
            buffer.clear()

    # Scriviamo gli ultimi dati rimasti nel buffer
    if buffer:
        random.shuffle(buffer)
        for X_seq, y_seq, partition in buffer:
            writers[f"X_{partition}"].writerow(X_seq)
            writers[f"y_{partition}"].writerow(y_seq)
            partition_sizes[partition] += 1

    # Chiudiamo i file CSV
    for file in file_handlers.values():
        file.close()

    # **Creiamo il file con le shape delle partizioni**
    shape_info = (
        f"Dataset salvato in {folder_path}\n"
        f"Dimensioni delle partizioni:\n"
        f"- X_train shape: ({partition_sizes['train']}, {len(X_seq)})\n"
        f"- y_train shape: ({partition_sizes['train']}, {len(y_seq)})\n"
        f"- X_val shape: ({partition_sizes['val']}, {len(X_seq)})\n"
        f"- y_val shape: ({partition_sizes['val']}, {len(y_seq)})\n"
        f"- X_test shape: ({partition_sizes['test']}, {len(X_seq)})\n"
        f"- y_test shape: ({partition_sizes['test']}, {len(y_seq)})\n"
    )

    with open(file_paths["shape_info"], "w") as shape_file:
        shape_file.write(shape_info)

    # Esempio di utilizzo
    save_config_to_txt(config, foldername)

    # Stampa delle dimensioni delle partizioni
    print(shape_info)

    return partition_sizes












