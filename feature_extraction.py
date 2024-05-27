"""
# Script for feature extraction of AE and spindle current measurements.
"""

import os
import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
import pywt

# Function to extract time-domain features
def extract_time_features(data):
    features = {
        'mean': np.mean(data),
        'std': np.std(data),
        'var': np.var(data),
        'skewness': skew(data),
        'kurtosis': kurtosis(data),
        'rms': np.sqrt(np.mean(data**2)),
        'peak_to_peak': np.ptp(data),
        'crest_factor': np.max(np.abs(data)) / np.sqrt(np.mean(data**2)),
        'median': np.median(data),
        'mode': pd.Series(data).mode().iloc[0] if len(pd.Series(data).mode()) > 0 else np.nan,
        'range': np.max(data) - np.min(data),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        'entropy': entropy(np.abs(data)),
        'zero_crossing_rate': ((data[:-1] * data[1:]) < 0).sum() / len(data),
        # 'autocorrelation': np.correlate(data, data, mode='full')[len(data)-1]
    }
    return features

# Function to extract frequency-domain features
def extract_frequency_features(data, sampling_rate):
    freqs, psd = welch(data, fs=sampling_rate)
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
    spectral_flatness = np.exp(np.mean(np.log(psd))) / np.mean(psd)
    spectral_rolloff = freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]]
    features = {
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_flatness': spectral_flatness,
        'spectral_rolloff': spectral_rolloff,
        'dominant_frequency': freqs[np.argmax(psd)],
        'mean_psd': np.mean(psd),
        'median_psd': np.median(psd),
        'max_psd': np.max(psd),
        'min_psd': np.min(psd)
    }
    return features

# Function to extract wavelet features
def extract_wavelet_features(data):
    coeffs = pywt.wavedec(data, 'db1', level=5)
    features = {}
    for i, coeff in enumerate(coeffs):
        features[f'wavelet_mean_{i}'] = np.mean(coeff)
        features[f'wavelet_std_{i}'] = np.std(coeff)
        features[f'wavelet_energy_{i}'] = np.sum(coeff**2)
    return features

# Function to aggregate all features
def extract_all_features(data, sampling_rate):
    time_features = extract_time_features(data)
    frequency_features = extract_frequency_features(data, sampling_rate)
    wavelet_features = extract_wavelet_features(data)
    all_features = {**time_features, **frequency_features, **wavelet_features}
    return all_features

#--------------------------------------------------------------

OK_DIRECTORY = '../Data/OK_Measurements'
all_features_list = []

# Iterate through each folder in the base directory
for folder_name in os.listdir(OK_DIRECTORY):
    folder_path = os.path.join(OK_DIRECTORY, folder_name)
    raw_path = os.path.join(folder_path, "raw")

    if os.path.isdir(raw_path):
        for file_name in os.listdir(raw_path):
            file_path = os.path.join(raw_path, file_name)
            if "2000KHz" in file_name:
                ae_data = pd.read_parquet(file_path)
            else:
                current_data = pd.read_parquet(file_path)

        features_ae = extract_all_features(ae_data.values.flatten(), sampling_rate=2000)
        features_rms = extract_all_features(
            current_data["Irms_Grinding_rate100000_clipping0_batch0"], sampling_rate=1000)
        features_li1 = extract_all_features(
            current_data["Grinding spindle current L1_rate100000_clipping0_batch0"],
            sampling_rate=1000)
        features_li2 = extract_all_features(
            current_data["Grinding spindle current L2_rate100000_clipping0_batch0"],
            sampling_rate=1000)
        features_li3 = extract_all_features(
            current_data["Grinding spindle current L3_rate100000_clipping0_batch0"],
            sampling_rate=1000)

        features_ae = {f"ae_{k}": v for k, v in features_ae.items()}
        features_rms = {f"rms_{k}": v for k, v in features_ae.items()}
        features_li1 = {f"li1_{k}": v for k, v in features_ae.items()}
        features_li2 = {f"li2_{k}": v for k, v in features_ae.items()}
        features_li3 = {f"li3_{k}": v for k, v in features_ae.items()}

        combined_features = {**features_ae, **features_rms, **features_li1, **features_li2,
                             **features_li3}
        combined_features["anomaly"] = False

        all_features_list.append(combined_features)

all_features_df = pd.DataFrame(all_features_list)
all_features_df.to_parquet('features.parquet', engine='pyarrow')
