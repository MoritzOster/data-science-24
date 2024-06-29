#!/usr/bin/env python3

"""
# Script for splitting OK and NOK data, upsampling using SMOTE, and saving to separate Parquet files.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Function to apply SMOTE for upsampling
def apply_smote(df, target_column):
    smote = SMOTE(random_state=42)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Fit SMOTE and generate synthetic samples
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Combine the resampled features and target into a single DataFrame
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[target_column])], axis=1)
    
    return df_resampled

def data_split():
    # Read the extracted features from the Parquet files
    nok_file = '../data/nok_features.parquet'
    ok_file = '../data/ok_features.parquet'
    df_nok = pd.read_parquet(nok_file)
    df_ok = pd.read_parquet(ok_file)

    # Limit the amount of OK data used to 500 samples
    df_ok_limited = df_ok.sample(n=500, random_state=42)

    # Combine OK and NOK data
    df_combined = pd.concat([df_ok_limited, df_nok], ignore_index=True)

    # Split the combined data into training and test sets (80:20)
    train_df, test_df = train_test_split(df_combined, test_size=0.2, stratify=df_combined['anomaly'], random_state=42)
    
    return train_df, test_df

def upsampling(train_df, test_df):
    # Perform SMOTE upsampling on the training data
    train_df_upsampled = apply_smote(train_df, 'anomaly')

    # Perform SMOTE upsampling on the test data
    test_df_upsampled = apply_smote(test_df, 'anomaly')

    # Write the original training and test datasets to new Parquet files
    train_output_file = '../data/train_features.parquet'
    test_output_file = '../data/test_features.parquet'
    train_df.to_parquet(train_output_file, engine='pyarrow', compression='snappy', index=False)
    test_df.to_parquet(test_output_file, engine='pyarrow', compression='snappy', index=False)

    print(f"Original training data written to {train_output_file}")
    print(f"Original test data written to {test_output_file}")

    # Write the upsampled training and test datasets to new Parquet files
    upsampled_train_output_file = '../data/upsampled_train_features.parquet'
    upsampled_test_output_file = '../data/upsampled_test_features.parquet'

    train_df_upsampled.to_parquet(upsampled_train_output_file, engine='pyarrow', compression='snappy')
    test_df_upsampled.to_parquet(upsampled_test_output_file, engine='pyarrow', compression='snappy')

    print(f"Upsampled training data written to {upsampled_train_output_file}")
    print(f"Upsampled test data written to {upsampled_test_output_file}")

    # Print class distribution to verify
    print("Original training class distribution:")
    print(train_df['anomaly'].value_counts())

    print("Upsampled training class distribution:")
    print(train_df_upsampled['anomaly'].value_counts())

    print("Original test class distribution:")
    print(test_df['anomaly'].value_counts())

    print("Upsampled test class distribution:")
    print(test_df_upsampled['anomaly'].value_counts())