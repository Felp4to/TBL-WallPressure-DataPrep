# time_features.py

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis



# Extract time features and optionally print results
def extract_time_features(window, show_result=False):
    # cluster 1
    mean_val = np.mean(window)                                       # mean
    std_val = np.std(window)                                         # standard deviation
    min_val = np.min(window)                                         # min
    max_val = np.max(window)                                         # max
    range_val = max_val - min_val                                    # range
    # cluster 2
    skew_val = skew(window)                                          # skewness
    kurtosis_val = kurtosis(window)                                  # kurtosis
    # cluster 3
    energy_val = np.sum(window ** 2)                                 # energy

    features = np.array([
        mean_val,
        std_val,
        min_val,
        max_val,
        range_val,
        skew_val,
        kurtosis_val,
        energy_val
    ])

    if show_result:
        print("Time Domain Features:")
        print("Cluster_1: ")
        print(f"\tMean: {mean_val}")
        print(f"\tStandard Deviation: {std_val}")
        print(f"\tMin: {min_val}")
        print(f"\tMax: {max_val}")
        print(f"\tRange: {range_val}")
        print("Cluster_2: ")
        print(f"\tSkewness: {skew_val}")
        print(f"\tKurtosis: {kurtosis_val}")
        print("Cluster_3: ")
        print(f"\tEnergy: {energy_val}")

    return features































