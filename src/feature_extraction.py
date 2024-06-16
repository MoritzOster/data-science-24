#!/usr/bin/env python3

"""
# Script for feature extraction of AE and spindle current measurements.
"""

import os
import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
import pyarrow as pa
import pyarrow.parquet as pq
import pywt
import zipfile
from io import BytesIO

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

def extract_all_combined_features(ae_data, current_data, anomaly):
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
    features_rms = {f"rms_{k}": v for k, v in features_rms.items()}
    features_li1 = {f"li1_{k}": v for k, v in features_li1.items()}
    features_li2 = {f"li2_{k}": v for k, v in features_li2.items()}
    features_li3 = {f"li3_{k}": v for k, v in features_li3.items()}

    combined_features = {**features_ae, **features_rms, **features_li1, **features_li2,
                            **features_li3}
    combined_features["anomaly"] = anomaly

    return combined_features

def extract_features_from_path(path, anomaly, output_file):
    first_write = True

    number_of_files = len(os.listdir(path))
    already_processed = 0

    # Create a ParquetWriter instance if writing for the first time
    writer = None

    # Iterate through each folder in the base directory
    for folder_name in os.listdir(path):
        print(f"Processing folder {already_processed} / {number_of_files} ({number_of_files - already_processed} remaining)")
        already_processed += 1
        folder_path = os.path.join(path, folder_name)
        raw_path = os.path.join(folder_path, "raw")

        if os.path.isdir(raw_path):
            ae_data = None
            current_data = None
            for file_name in os.listdir(raw_path):
                file_path = os.path.join(raw_path, file_name)
                if "2000KHz" in file_name:
                    ae_data = pd.read_parquet(file_path)
                else:
                    current_data = pd.read_parquet(file_path)

            if ae_data is not None and current_data is not None:
                combined_features = extract_all_combined_features(ae_data, current_data, anomaly)
                df = pd.DataFrame([combined_features])
                table = pa.Table.from_pandas(df)
                
                if first_write:
                    writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
                    first_write = False
                
                writer.write_table(table)
    
    if writer:
        writer.close()

def extract_features_from_path_cluster(path, anomaly, output_file):
    first_write = True

    # Create a ParquetWriter instance if writing for the first time
    writer = None

    for file in os.listdir(path):
        zip_path = os.path.join(path, file)
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if name.endswith("Grinding/"):
                    raw_path = os.path.join(name, "raw/")
                    target_contents = [item for item in z.namelist() if item.startswith(raw_path)]

                    ae_data = None
                    current_data = None
                    for file_path in target_contents:
                        with z.open(file_path) as file:
                            if "2000KHz" in file_path:
                                buffer = BytesIO(file.read())
                                ae_data = pd.read_parquet(buffer)
                            elif "100KHz" in file_path:
                                buffer = BytesIO(file.read())
                                current_data = pd.read_parquet(buffer)
            
                    if ae_data is not None and current_data is not None:
                        combined_features = extract_all_combined_features(ae_data, current_data, anomaly)
                        df = pd.DataFrame([combined_features])
                        table = pa.Table.from_pandas(df)
                        
                        if first_write:
                            writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
                            first_write = False
                        
                        writer.write_table(table)
    
    if writer:
        writer.close()

#--------------------------------------------------------------

OK_DIRECTORY = '/home/dsbwl24_team001/data'
extract_features_from_path_cluster(OK_DIRECTORY, False, 'ok_features.parquet')


# OK_DIRECTORY = '/home/dsbwl24_team001/data'
# OK_DIRECTORY = '../../Test_202402-4/'
# all_features_df_ok = extract_features_from_path_cluster(OK_DIRECTORY, False)
# extract_features_from_path(OK_DIRECTORY, False, 'ok_features.parquet')
# all_features_df_ok.to_parquet('ok_features.parquet', engine='pyarrow')

# NOK_DIRECTORY = '../../Data/NOK_Measurements'
# extract_features_from_path(NOK_DIRECTORY, True, 'nok_features.parquet')