# signal_statistics.py

import numpy as np




def max_min(signal):
    return np.max(signal), np.min(signal)

def mean_median(signal):
    return np.mean(signal), np.median(signal)

def variance(signal):
    return np.var(signal)

def std_deviation(signal):
    return np.std(signal)

def energy(signal):
    return np.sum(np.abs(signal) ** 2)

def power(signal):
    return energy(signal) / len(signal)

def mean_power(signal):
    return np.mean(signal ** 2)

def range_amplitude(signal):
    max_val, min_val = max_min(signal)
    return max_val - min_val

def zero_crossings(signal):
    return np.sum(np.diff(np.sign(signal)) != 0)

def print_all_statistics(signal):
    max_val, min_val = max_min(signal)
    mean_val, median_val = mean_median(signal)
    var_val = variance(signal)
    std_val = std_deviation(signal)
    energy_val = energy(signal)
    power_val = power(signal)
    mean_power_val = mean_power(signal)
    range_val = range_amplitude(signal)
    zero_cross = zero_crossings(signal)

    print(f"Max: {max_val:.4f}")
    print(f"Min: {min_val:.4f}")
    print(f"Range: {range_val:.4f}")
    print(f"Mean/Median: {mean_val:.4f}")
    print(f"Median: {median_val:.4f}")
    print(f"Variance: {var_val:.4f}")
    print(f"Standard Deviation: {std_val:.4f}")
    print(f"Energy: {energy_val:.4f}")
    print(f"Power: {power_val:.4f}")
    print(f"Mean Power: {mean_power_val:.4f}")
    print(f"Zero Crossings: {zero_cross}")

