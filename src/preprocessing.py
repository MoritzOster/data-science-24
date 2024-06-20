#!/usr/bin/env python3

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load(path):
    return pd.read_parquet(path)

def remove_target(df):
    X_df = df.drop(columns=['anomaly'])
    y_df = df['anomaly']

    return X_df, y_df

def remove_correlated_features(X_train, X_test):
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)

    return X_train, X_test

def plot_correlation(X_df, y_df):
    test = X_df
    test['anomaly'] = y_df
    plt.figure(figsize=(12,12))
    cor = test.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()

def scale_data(X_train, X_test):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = X_train.values
    X_test = X_test.values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test

def perform_pca(X_train, X_test):
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test

def pca_plot(path_train, path_test):
    X_train, y_train, X_test, y_test = preprocess(path_train, path_test)
    # Convert to DataFrame for easy plotting
    train_df = pd.DataFrame(data=X_train, columns=['principal component 1', 'principal component 2'])
    test_df = pd.DataFrame(data=X_test, columns=['principal component 1', 'principal component 2'])

    if y_train is not None:
        train_df['label'] = y_train
    if y_test is not None:
        test_df['label'] = y_test

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot training data
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='principal component 1', y='principal component 2', hue='label', data=train_df, palette='viridis')
    plt.title('PCA of Training Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Label')
    plt.grid()

    # Plot test data
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='principal component 1', y='principal component 2', hue='label', data=test_df, palette='viridis')
    plt.title('PCA of Test Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Label')
    plt.grid()

    plt.show()

#------------------------------------------------------

def preprocess(path_train, path_test):
    train_df = load(path_train)
    test_df = load(path_test)

    X_test, y_test = remove_target(test_df)
    X_train, y_train = remove_target(train_df)

    X_train, X_test = remove_correlated_features(X_train, X_test)

    #plot_correlation(X_train, y_train)

    X_train, X_test = scale_data(X_train, X_test)

    X_train, X_test = perform_pca(X_train, X_test)

    return X_train, y_train, X_test, y_test