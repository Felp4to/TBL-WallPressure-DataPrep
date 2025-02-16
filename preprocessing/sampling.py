# sampling.py

import pandas as pd


# sampling standard
def sampling_csv(file_path, sampling_num):
    df = pd.read_csv(file_path)
    return df.iloc[::sampling_num]

# sampling with average
def sampling_avg_csv(file_path, sampling_num):
    df = pd.read_csv(file_path)
    df_reduced = df['singleData'].groupby(df.index // sampling_num).mean().reset_index(drop=True)
    return df_reduced