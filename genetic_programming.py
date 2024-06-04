import numpy as np
from tpot import TPOTClassifier
from generate_data import *
from sklearn.model_selection import train_test_split
import json

data_path = './features.parquet'
data = pd.read_parquet(data_path)


X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data['anomaly'],
                                                    train_size=0.8, test_size=0.2)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train.values
X_test = X_test.values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


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

pipeline_optimizer = TPOTClassifier(config_dict=tpot_config, generations=5, population_size=20, cv=10, random_state=42, verbosity=2)

pipeline_optimizer.fit(X_train, y_train)

print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('best_pipeline.py')



