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

# Read the extracted features from the Parquet files
nok_file = 'nok_features.parquet'
ok_file = 'ok_features.parquet'
df_nok = pd.read_parquet(nok_file)
df_ok = pd.read_parquet(ok_file)

# Limit the amount of OK data used to 1000 samples
df_ok_limited = df_ok.sample(n=1000, random_state=42)

# Combine OK and NOK data
df_combined = pd.concat([df_ok_limited, df_nok], ignore_index=True)

# Split the combined data into training and test sets (80:20)
train_df, test_df = train_test_split(df_combined, test_size=0.2, stratify=df_combined['anomaly'], random_state=42)

# Perform SMOTE upsampling on the training data
train_df_upsampled = apply_smote(train_df, 'anomaly')

# Perform SMOTE upsampling on the test data
test_df_upsampled = apply_smote(test_df, 'anomaly')

# Write the upsampled training and test datasets to new Parquet files
train_output_file = 'upsampled_train_features.parquet'
test_output_file = 'upsampled_test_features.parquet'
train_df_upsampled.to_parquet(train_output_file, engine='pyarrow', compression='snappy')
test_df_upsampled.to_parquet(test_output_file, engine='pyarrow', compression='snappy')

print(f"Upsampled training data written to {train_output_file}")
print(f"Upsampled test data written to {test_output_file}")

# Print class distribution to verify
print("Original training class distribution:")
print(train_df['anomaly'].value_counts())

print("Upsampled training class distribution:")
print(train_df_upsampled['anomaly'].value_counts())

print("Original test class distribution:")
print(test_df['anomaly'].value_counts())

print("Upsampled test class distribution:")
print(test_df_upsampled['anomaly'].value_counts())