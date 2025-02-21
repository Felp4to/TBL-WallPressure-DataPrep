# data_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def estimate_period_from_fft(filepath, interval):
    df = pd.read_csv(filepath)                                  # Carica il dataset
    data = df["singleData"].head(interval).values               # Seleziona i primi 'interval' dati
    fft_trasformata = np.fft.fft(data)                          # Calcolo della Trasformata di Fourier Discreta (FFT)
    frequenze = np.fft.fftfreq(len(data))                       # Calcolo delle frequenze corrispondenti
    plt.figure(figsize=(20, 10))                                # Grafico della trasformata (modulo dello spettro)
    plt.plot(frequenze, np.abs(fft_trasformata))
    plt.title("Spettro della Trasformata di Fourier")
    plt.xlabel("Frequenza (Hz)")
    plt.ylabel("Ampiezza")
    plt.grid()
    plt.show()


def plot_timeseries(filepath, interval):
    df = pd.read_csv(filepath)                                  # Carica il dataset
    data = df["singleData"].head(interval)                      # Seleziona i primi 'interval' dati
    plt.figure(figsize=(20, 10))                                # Creazione del grafico
    plt.plot(data.index, data, marker='o',
              markersize=4, linestyle='-',
                color='#3498DB', label="Time Series")
    plt.xlabel("Index")                                         # Mostrare il grafico
    plt.ylabel("Value")
    plt.title("Time Series Plot")
    plt.legend()
    plt.grid()
    plt.show()




