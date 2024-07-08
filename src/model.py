import joblib
import numpy as np
import pandas as pd
from preprocessing import *
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import *
from evaluation import *

path_train = '../data/upsampled_train_features.parquet'
path_test = '../data/test_features.parquet'

X_train, y_train, X_test, y_test = preprocess(path_train, path_test)

# Train a logistic regression and LDA classifier on the PCA-reduced data
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)

lda_classifier = LinearDiscriminantAnalysis()
lda_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr = lr_classifier.predict(X_test)
y_pred_lda = lda_classifier.predict(X_test)

# Save the trained model to a file
with open('../data/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_classifier, f)
    
results = evaluate_model(lr_classifier, X_train, y_train)
print(results)
results = evaluate_model(lr_classifier, X_test, y_test)
print(results)
plot_classifier_results(lr_classifier, X_test, y_test, 'Logistic_Regression')


