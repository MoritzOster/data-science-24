from sklearn.svm import SVC
from tpot import TPOTClassifier
from generate_data import *
from sklearn.model_selection import train_test_split
import json


data_path = '/Users/emilyries/Downloads/Data_Science/Project/Implementation/data-science-24/features.parquet'
data = pd.read_parquet(data_path)
#spindle_data = data.loc[:,~data.columns.str.startswith('ae')]


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
        'estimator': [SVC(kernel='linear')],
        'n_features_to_select': [0.2, 0.4, 0.6, 0.8, 1.0]
    },
    'sklearn.feature_selection.RFE': {
        'estimator': [SVC(kernel='rbf')],
        'n_features_to_select': [0.2, 0.4, 0.6, 0.8, 1.0]
    }
}

pipeline_optimizer = TPOTClassifier(config_dict=tpot_config, generations=5, population_size=20, cv=10, random_state=42, verbosity=2)

pipeline_optimizer.fit(X_train, y_train)
evaluated_individuals = json.dumps(pipeline_optimizer.evaluated_individuals_, indent=2)
with open('evaluated_individuals.json', 'a') as j:
    j.write(evaluated_individuals)


print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('best_pipeline.py')



