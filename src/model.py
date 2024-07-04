import joblib
import numpy as np
import pandas as pd
from preprocessing import *
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import *

path_train = '../data/train_features.parquet'
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
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_classifier, f)
    
# Save the trained model to a file
with open('lda_model.pkl', 'wb') as f:
    pickle.dump(lda_classifier, f)




