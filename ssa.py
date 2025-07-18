
import numpy as np

def embed_series(series, window_size):
    N = len(series)
    K = N - window_size + 1
    trajectory_matrix = np.column_stack([series[i:i + K] for i in range(window_size)])
    return trajectory_matrix

def ssa_decompose(series, window_size):
    X = embed_series(series, window_size)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U, S, Vt, X

def reconstruct_component(U, S, Vt, components):
    X_elem = np.array([S[i] * np.outer(U[:, i], Vt[i, :]) for i in components])
    X_reconstructed = np.sum(X_elem, axis=0)
    return diagonal_averaging(X_reconstructed)

def diagonal_averaging(X):
    L, K = X.shape
    N = L + K - 1
    result = np.zeros(N)
    count = np.zeros(N)

    for i in range(L):
        for j in range(K):
            result[i + j] += X[i, j]
            count[i + j] += 1

    return result / count
