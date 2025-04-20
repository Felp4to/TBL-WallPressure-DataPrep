# normalizzation.py

# here are defined methods with the purpose to normalize a dataframe of pressure values
import numpy as np



# normalization balanced
def normalize_array(arr):
    max_abs_value = np.max(np.abs(arr))
    
    if max_abs_value == 0:
        normalized = np.zeros_like(arr)
    else:
        normalized = arr / max_abs_value
    
    return normalized 

# denormalize
def denormalize_array(normalized_arr, max_abs_value):
    return normalized_arr * max_abs_value


