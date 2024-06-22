#!/usr/bin/env python3

from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


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

# Function to evaluate all baseline models + ProDetect
def evaluate(X_train, y_train, X_test, y_test, X_train_oc, y_train_oc, X_test_oc, y_test_oc):
    # Define baseline models
    KNC_model = KNeighborsClassifier(n_neighbors=9)
    SVC_model = SVC(C=10, kernel="linear")
    RFC_model = RandomForestClassifier(max_features="log2", min_samples_leaf=1, min_samples_split=10, n_estimators=200)
    ETC_model = ExtraTreesClassifier(max_features="log2", min_samples_leaf=4, min_samples_split=10, n_estimators=100)
    OC_SVM_model = OneClassSVM(gamma=0.01, nu=0.1)

    # Fit baseline models
    KNC_model.fit(X_train, y_train)
    SVC_model.fit(X_train, y_train)
    RFC_model.fit(X_train, y_train)
    ETC_model.fit(X_train, y_train)
    OC_SVM_model.fit(X_train_oc, y_train_oc)

    models = {
        'KNC': KNC_model,
        'SVC': SVC_model,
        'RFC': RFC_model,
        'ETC': ETC_model,
        'OC_SVM': OC_SVM_model
    }

    # Evaluate all models
    results = {}
    for name, model in models.items():
        if name == 'OC_SVM':
            results[name] = evaluate_model(model, X_test_oc, y_test_oc)
        else:
            results[name] = evaluate_model(model, X_test, y_test)

    # Print results
    for name, metrics in results.items():
        print(f"Results for {name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print("\n" + "-"*60 + "\n")
