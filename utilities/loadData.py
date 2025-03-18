# loadData.py

import pandas as pd



# load time series from file csv
def load_time_series(file_path):
    df = pd.read_csv(file_path)
    return df["singleData"].values

# load dataframe from file csv
def load_dataframe(file_path):
    return pd.read_csv(file_path)
























