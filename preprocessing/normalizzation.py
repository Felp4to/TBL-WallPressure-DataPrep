# normalizzation.py

# here are defined methods with the purpose to normalize a dataframe of pressure values


# Max-Min Scaling method
def max_min_scaling_normalizzation(df):
    df['singleData'] = (df['singleData'] - df['singleData'].min()) / (df['singleData'].max() - df['singleData'].min())
    return df  # Restituiamo l'intero DataFrame per mantenere il nome della colonna

# Z-Score method
def z_score_normalizzation(df):
    df['singleData'] = (df['singleData'] - df['singleData'].mean()) / df['singleData'].std()
    return df  # Restituiamo l'intero DataFrame per mantenere il nome della colonna
