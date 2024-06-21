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
        train_df['anomaly'] = y_train
    if y_test is not None:
        test_df['anomaly'] = y_test


    plt.rc ('font', size = 14)
    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot training data
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='principal component 1', y='principal component 2', hue='anomaly', data=train_df)
    plt.title('PCA of Training Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Anomaly')
    plt.grid()

    # Plot test data
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='principal component 1', y='principal component 2', hue='anomaly', data=test_df)
    plt.title('PCA of Test Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Anomaly')
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

def one_class_preprocess(ok_path, nok_path):
    ok_data = load(ok_path)
    nok_data = load(nok_path)
    
    ok_data, y_ok = remove_target(ok_data)
    nok_data, y_nok = remove_target(nok_data)
    
    X_train = ok_data.iloc[:int(len(ok_data)*0.8)]
    y_train = y_ok.iloc[:int(len(y_ok)*0.8)]
    
    ok_eval_data = ok_data.iloc[int(len(ok_data)*0.8):]
    ok_eval_y = y_ok.iloc[int(len(y_ok)*0.8):]
    
    X_test = pd.concat([ok_eval_data, nok_data], axis=0)
    y_test = pd.concat([ok_eval_y, y_nok], axis=0)
    
    X_train, X_test = remove_correlated_features(X_train, X_test)
    
    X_train, X_test = scale_data(X_train, X_test)

    X_train, X_test = perform_pca(X_train, X_test)
    
    y_test = y_test.apply(lambda x : {True: -1, False: 1}.get(x))
    y_train = y_train.apply(lambda x: {True: -1, False: 1}.get(x))
    
    return X_train, y_train, X_test, y_test
    
    

# pca_plot('./train_features.parquet', './test_features.parquet')