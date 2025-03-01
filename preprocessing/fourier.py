# fourier.py

from tqdm import tqdm
import pandas as pd



def prova(df, window_size):

    # convert dataframe into a flattened array
    timeseries = df.to_numpy().flatten()

    # calculate number of windows
    num_windows = len(timeseries) // window_size
    num_windows = 100

    # 
    for i in range(0, num_windows):
        print(timeseries[window_size * i : window_size * i + window_size])
    
    # aaa
    # windows = [timeseries[i:i+window_size] for i in range(0, len(timeseries) - window_size + 1, window_size)]


    
    return timeseries


def prova2(df, window_size):
    windows = []  # Lista per memorizzare le finestre

    # Usiamo tqdm per la progress bar
    for i in tqdm(range(0, len(df) - window_size + 1, window_size), desc="Elaborazione finestre"):
        windows.append(df['singleData'].iloc[i : i + window_size].values)

    # Creiamo il DataFrame finale
    df_transformed = pd.DataFrame(windows, columns=[f'dp[{i}]' for i in range(window_size)])
    
    return df_transformed
















