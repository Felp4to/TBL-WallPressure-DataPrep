# analysis.py

import pandas as pd
import numpy as np



# Funzione per calcolare e stampare la media
def calcola_media(series):
    media = series.mean()
    print(f"Media: {media:.2f}")
    return media

# Funzione per calcolare e stampare la varianza
def calcola_varianza(series):
    varianza = series.var()
    print(f"Varianza: {varianza:.2f}")
    return varianza

# Funzione per calcolare e stampare la deviazione standard
def calcola_dev_std(series):
    dev_std = series.std()
    print(f"Deviazione standard: {dev_std:.2f}")
    return dev_std

# Funzione per calcolare e stampare il massimo
def calcola_massimo(series):
    massimo = series.max()
    print(f"Massimo: {massimo:.2f}")
    return massimo

# Funzione per calcolare e stampare il minimo
def calcola_minimo(series):
    minimo = series.min()
    print(f"Minimo: {minimo:.2f}")
    return minimo

# Funzione per calcolare e stampare il range (massimo - minimo)
def calcola_range(series):
    range_val = series.max() - series.min()
    print(f"Range: {range_val:.2f}")
    return range_val

# Funzione per calcolare e stampare la mediana
def calcola_mediana(series):
    if isinstance(series, np.ndarray):  
        series = series.flatten() 
        series = pd.Series(series)
    mediana = series.median()
    print(f"Mediana: {mediana:.2f}")
    return mediana

# Funzione per calcolare e stampare la potenza del segnale (array con x^2) (signal power)
def calcola_potenza_segnale(series):
    if isinstance(series, np.ndarray):  
        series = series.flatten()  # Assicura che sia 1D
        series = pd.Series(series)
    potenza_segnale = series ** 2
    print("Potenza del segnale calcolata (primi 5 valori):", potenza_segnale.head())
    return potenza_segnale

# Funzione per calcolare e stampare la potenza media (average power)
def calcola_potenza_media(series):
    if isinstance(series, np.ndarray):  
        series = series.flatten()  # Assicura che sia 1D
        series = pd.Series(series)
    potenza_media = (series ** 2).mean()
    print(f"Potenza media: {potenza_media:.2f}")
    return potenza_media



