# sampling.py

import constants as cs
import pandas as pd



# extract first n seconds from time series
def slicing(file_path, secs):
    """Estrae i primi n secondi della serie temporale."""
    df = pd.read_csv(file_path)
    num_samples = int(secs * cs.SAMPLING_RATE)
    return df[:num_samples]

# sampling standard
def sampling_csv(data_df, sampling_factor):
    #df = pd.read_csv(file_path)
    #print_memory_usage(data_df)
    return data_df.iloc[::sampling_factor]


# sampling with average
def sampling_avg_csv(data_df, sampling_factor):
    #df = pd.read_csv(file_path)
    df_reduced = data_df['singleData'].groupby(data_df.index // sampling_factor).mean().reset_index(drop=True)
    #print_memory_usage(df_reduced)
    return df_reduced


def print_memory_usage(data):
    # Calcola la memoria totale
    total_memory_gb = data.memory_usage(deep=True).sum() / (1024 ** 3)
    print(f"La memoria totale occupata dal DataFrame è: {total_memory_gb:.2f} GB")