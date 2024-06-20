#!/usr/bin/env python3

from preprocessing import preprocess
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

X_train, y_train, X_test, y_test = preprocess('./upsampled_train_features.parquet', './test_features.parquet')

# Define baseline models
SVC_model = SVC(C=10, kernel="linear")
KNC_model = KNeighborsClassifier(n_neighbors=9)
RFC_model = RandomForestClassifier(max_features="log2", min_samples_leaf=4, min_samples_split=2, n_estimators=200)
ETC_model = ExtraTreesClassifier(max_features="log2", min_samples_leaf=4, min_samples_split=2, n_estimators=200)

# Fit baseline models
SVC_model.fit(X_train, y_train)
KNC_model.fit(X_train, y_train)
RFC_model.fit(X_train, y_train)
ETC_model.fit(X_train, y_train)

models = {
    'SVC': SVC_model,
    'KNC': KNC_model,
    'RFC': RFC_model,
    'ETC': ETC_model
}

# Function to evaluate a single model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc,
    }

# Evaluate all models
results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X_train, y_train)

# Print results
for name, metrics in results.items():
    print(f"Results for {name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print("\n" + "-"*60 + "\n")
