#!/usr/bin/env python3

import pickle
import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import pywt

def extract_relevant_features(data, sampling_rate):
    freqs, psd = welch(data, fs=sampling_rate)
    features = {
        'mean': np.mean(data),
        'std': np.std(data),
        'skewness': skew(data),
        'kurtosis': kurtosis(data),
        'crest_factor': np.max(np.abs(data)) / np.sqrt(np.mean(data**2)),
        'median': np.median(data),
        'mode': pd.Series(data).mode().iloc[0] if len(pd.Series(data).mode()) > 0 else np.nan,
        'zero_crossing_rate': ((data[:-1] * data[1:]) < 0).sum() / len(data),
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
    return extract_relevant_features(data, sampling_rate=2000)

def prodetect_predict(path):
    scaler = pickle.load(open("../data/scaler.pkl", "rb"))
    pca = pickle.load(open("../data/pca_model.pkl", "rb"))
    prodetect = pickle.load(open("../data/best_model.pkl", "rb"))
    features = extract_features_from_file(path)
    del features["wavelet_mean_0"]
    feature_values = np.array(list(features.values())).reshape(1, -1)
    scaled_features = scaler.transform(feature_values)
    input = pca.transform(scaled_features)
    prediction = prodetect.predict(input)
    return prediction
