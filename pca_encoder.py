
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd

def apply_scaling(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data), scaler

def apply_pca(features, n_components=3):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(features)
    return transformed, pca

def encode_labels(labels):
    encoder = OneHotEncoder(sparse=False)
    return encoder.fit_transform(labels.reshape(-1, 1)), encoder
