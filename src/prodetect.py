#!/usr/bin/env python3

import pickle
import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
import pywt

def extract_relevant_features(data, sampling_rate):
    freqs, psd = welch(data, fs=sampling_rate)
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
    spectral_rolloff = freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]]
    #TODO: check again whether these are actually the relevant features
    features = {
        'mean': np.mean(data),
        'std': np.std(data),
        'skewness': skew(data),
        'kurtosis': kurtosis(data),
        'peak_to_peak': np.ptp(data),
        'crest_factor': np.max(np.abs(data)) / np.sqrt(np.mean(data**2)),
        'median': np.median(data),
        'mode': pd.Series(data).mode().iloc[0] if len(pd.Series(data).mode()) > 0 else np.nan,
        'entropy': entropy(np.abs(data)),
        'zero_crossing_rate': ((data[:-1] * data[1:]) < 0).sum() / len(data),
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_rolloff': spectral_rolloff,
        'dominant_frequency': freqs[np.argmax(psd)],
        'median_psd': np.median(psd),
        'min_psd': np.min(psd),
    }
    coeffs = pywt.wavedec(data, 'db1', level=5)
    for i, coeff in enumerate(coeffs):
        features[f'wavelet_mean_{i}'] = np.mean(coeff)
    return features

def extract_features_from_file(path):
    df = pd.read_parquet(path)
    data = df.values.flatten()
    return extract_relevant_features(data, sampling_rate=20000)

def prodetect_predict(path):
    pca = pickle.load(open("../data/pca_model.pkl", "rb"))
    prodetect = pickle.load(open("../data/best_model.pkl", "rb"))
    features = extract_features_from_file(path)
    del features["wavelet_mean_0"]
    feature_values = np.array(list(features.values())).reshape(1, -1)
    input = pca.transform(feature_values)
    print(prodetect.predict(input))
