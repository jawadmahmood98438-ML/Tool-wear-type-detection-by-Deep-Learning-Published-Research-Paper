
import pandas as pd

def load_force_data(filepath):
    return pd.read_csv(filepath)

def reshape_for_lstm(X):
    return X.reshape((X.shape[0], X.shape[1], 1))
