#!/usr/bin/env python3

from feature_extraction import extract_features_from_path
from data_split import data_split, upsampling
from preprocessing import preprocess, one_class_preprocess
from genetic_programming import genetic_programming
from evaluation import evaluate

def pipeline(ok_directory, nok_directory, run_genetic_programming):

    #Extract Features from the data
    if ok_directory != '':
        extract_features_from_path(ok_directory, False, '../data/ok_features.parquet')
    if nok_directory != '':
        extract_features_from_path(nok_directory, True, '../data/nok_features.parquet')

    #Split the data / define the split ratio
    train_df, test_df = data_split()

    #Upsample the NOK Data to balance the dataset
    upsampling(train_df, test_df)

    #Preprocessing stepss
    X_train, y_train, X_test, y_test = preprocess('../data/upsampled_train_features.parquet', '../data/test_features.parquet')
    X_train_oc, y_train_oc, X_test_oc, y_test_oc = one_class_preprocess('../data/ok_features.parquet', '../data/nok_features.parquet')

    #Use TPOT - genetic Programming to find the best model + export it
    # if run_genetic_programming:
    #     genetic_programming(X_train, y_train)

    #Evaluate the model
    evaluate(X_train, y_train, X_test, y_test, X_train_oc, y_train_oc, X_test_oc, y_test_oc)

pipeline('', '', False)
