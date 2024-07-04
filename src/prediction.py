#!/usr/bin/env python3

import pandas as pd
import pickle
import numpy as np

def extract_ae_std(data):
    return {
        'ae_std': np.std(data)
    }

def extract_features_from_file(ae_path):
    ae_df = pd.read_parquet(ae_path)
    ae_data = ae_df.values.flatten()
    return extract_ae_std(ae_data)

def predict(ae_path):
    scaler = pickle.load(open("../data/scaler.pkl", "rb"))
    model = pickle.load(open("../data/logistic_regression_model.pkl", "rb"))
    features = extract_features_from_file(ae_path)
    feature_values = np.array(list(features.values())).reshape(1, -1)
    scaled_features = scaler.transform(feature_values)
    print(model.predict(scaled_features)) 
