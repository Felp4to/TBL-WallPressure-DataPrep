# sampling.py

import pandas as pd



# sampling standard
def sampling_csv(file_path, sampling_num):
    df = pd.read_csv(file_path)
    print_memory_usage(df)
    return df.iloc[::sampling_num]


# sampling with average
def sampling_avg_csv(file_path, sampling_num):
    df = pd.read_csv(file_path)
    df_reduced = df['singleData'].groupby(df.index // sampling_num).mean().reset_index(drop=True)
    print_memory_usage(df_reduced)
    return df_reduced


def print_memory_usage(data):
    # Calcola la memoria totale
    total_memory_gb = data.memory_usage(deep=True).sum() / (1024 ** 3)
    print(f"La memoria totale occupata dal DataFrame è: {total_memory_gb:.2f} GB")