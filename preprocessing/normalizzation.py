# normalizzation.py

# here are defined methods with the purpose to normalize a dataframe of pressure values
import numpy as np



# Max-Min Scaling method
def max_min_scaling_normalizzation(df):
    df['singleData'] = (df['singleData'] - df['singleData'].min()) / (df['singleData'].max() - df['singleData'].min())
    return df  # Restituiamo l'intero DataFrame per mantenere il nome della colonna

# Z-Score method
def z_score_normalizzation(df):
    df['singleData'] = (df['singleData'] - df['singleData'].mean()) / df['singleData'].std()
    return df  # Restituiamo l'intero DataFrame per mantenere il nome della colonna

# normalization balanced
def normalize_dataframe(df):
    mean_value = df['singleData'].mean()                            # Calcola la media
    centered_series = df['singleData'] - mean_value                 # Centra la serie attorno a 0
    max_abs_value = centered_series.abs().max()                     # Trova il massimo valore assoluto
    if max_abs_value != 0:
        df['singleData'] = centered_series / max_abs_value          # Normalizza tra -1 e 1
    else:
        df['singleData'] = 0                                        # Evita divisione per zero
    return df