#!/usr/bin/env python3

import pickle
import numpy as np
import json
from tpot import TPOTClassifier
from preprocessing import preprocess

tpot_config = {
    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7, 9]
    },
    'sklearn.svm.SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'sklearn.feature_selection.SelectPercentile': {
        'percentile': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    },
    'sklearn.feature_selection.RFE': {
        'n_features_to_select': [0.2, 0.4, 0.6, 0.8, 1.0],
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }
}

def genetic_programming(X_train, y_train):
    pipeline_optimizer = TPOTClassifier(config_dict=tpot_config, generations=5, population_size=20, cv=10, random_state=42, verbosity=2)

    pipeline_optimizer.fit(X_train, y_train)

    pipeline_optimizer.export('model.py')
    model = pipeline_optimizer.fitted_pipeline_.steps[-1][1]

    with open('../data/best_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # evaluated_pipelines = pipeline_optimizer.evaluated_individuals_

    # with open('evaluated_pipelines.json', 'w') as json_file:
    #     json.dump(evaluated_pipelines, json_file, indent=4)

# X_train, y_train, X_test, y_test = preprocess('../data/upsampled_train_features.parquet', '../data/test_features.parquet')
# genetic_programming(X_train, y_train)
