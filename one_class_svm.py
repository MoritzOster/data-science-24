from generate_data import *
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, classification_report

data_path = '/Users/emilyries/Downloads/Data_Science/Project/Implementation/data-science-24/features.parquet'

train_data, test_data, ok_eval_data = get_data(data_path)

# Define the parameter grid
param_grid = {
    'nu': [0.1, 0.3, 0.5, 0.7, 0.9],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
}

# Define a custom scoring function
def custom_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1)  # -1 is considered as the anomaly class

# Create a scorer
scorer = make_scorer(custom_f1_score)

o_svm = OneClassSVM()
# Perform grid search
grid_search = GridSearchCV(o_svm, param_grid, refit=True, scoring=scorer, verbose=2, cv=5)
grid_search.fit(train_data)

# Print the best parameters and the best estimator
print("Best parameters found: ", grid_search.best_params_)
print("Best estimator found: ", grid_search.best_estimator_)

# Predict on the test set and outliers
y_pred_test = grid_search.predict(ok_eval_data)
y_pred_outliers = grid_search.predict(test_data)

# Print results
print("Prediction on test data:")
print(y_pred_test)

print("Prediction on outliers:")
print(y_pred_outliers)

print("Classification Report for Test Data:")
print(classification_report([1]*len(y_pred_test), y_pred_test))

print("Classification Report for Outliers:")
print(classification_report([-1]*len(y_pred_outliers), y_pred_outliers))

print('*' * 40)

# Train the One-Class SVM model
model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
model.fit(train_data)

y_pred_train = model.predict(train_data)
# Predict the test data
y_pred_test = model.predict(ok_eval_data)
# Predict the outliers
y_pred_outliers = model.predict(test_data)

# Print results
print("Prediction on training data:")
print(y_pred_train)

print("Prediction on test data:")
print(y_pred_test)

print("Prediction on outliers:")
print(y_pred_outliers)
