
import numpy as np
from scipy.stats import kurtosis, skew

def extract_features(signal):
    return {
        "mean": np.mean(signal),
        "absolute_mean": np.mean(np.abs(signal)),
        "std": np.std(signal),
        "var": np.var(signal),
        "max": np.max(signal),
        "min": np.min(signal),
        "kurtosis": kurtosis(signal),
        "skewness": skew(signal),
        "rms": np.sqrt(np.mean(signal**2)),
        "waveform_index": np.sqrt(np.mean(signal**2)) / np.mean(np.abs(signal)),
        "margin_index": np.max(np.abs(signal)) / np.mean(np.abs(signal)),
        "peak_index": np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
        "kurtosis_pulse_index": kurtosis(signal) / np.max(np.abs(signal)),
        "peak_to_peak": np.ptp(signal),
        "rms_amplitude": np.sqrt(np.sum(signal**2)/len(signal))
    }
