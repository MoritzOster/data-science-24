#from generate_data import *
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, classification_report, auc
from sklearn.preprocessing import StandardScaler
from evaluation import *

ok_data = pd.read_parquet('./ok_features.parquet').drop(columns= ['anomaly'])
test_data = pd.read_parquet('./nok_features.parquet').drop(columns= ['anomaly'])

train_data = ok_data.iloc[:int(len(ok_data)*0.8)]
ok_eval_data = ok_data.iloc[int(len(ok_data)*0.8):]

train_data = train_data.astype('float32').values
test_data = test_data.astype('float32').values
ok_eval_data = ok_eval_data.astype('float32').values

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)
ok_eval_data = scaler.transform(ok_eval_data)

# Define the parameter grid
param_grid = {
    'nu': [0.1, 0.3, 0.5, 0.7, 0.9],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
}

# Define a custom scoring function
def custom_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=-1)  # -1 is considered as the anomaly class

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

print("Classification Report for OK Test Data:")
print(classification_report([1]*len(y_pred_test), y_pred_test))

print("Classification Report for NOK Data:")
print(classification_report([-1]*len(y_pred_outliers), y_pred_outliers))


eval_x = np.concatenate((ok_eval_data, test_data), axis=0)
eval_y = np.concatenate(([1]*len(ok_eval_data), [-1]*len(test_data)), axis=0) 

plot_confusion_matrix(grid_search.best_estimator_, eval_x, eval_y)

plot_roc(grid_search.best_estimator_, eval_x, eval_y)
