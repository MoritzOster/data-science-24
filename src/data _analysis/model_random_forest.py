import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load OK data
ok_data1 = pd.read_csv('merged_spindle_ok.csv')
ok_data2 = pd.read_csv('merged_aeki_ok.csv')


# Combine OK data along columns, drop duplicate 'Time' column
ok_combined = pd.concat([ok_data1, ok_data2.drop(columns=['Time'])], axis=1)

# Load NOK data
nok_data1 = pd.read_csv('merged_spindle_nok.csv')
nok_data2 = pd.read_csv('merged_aeki_nok.csv')

# Combine NOK data along columns, drop duplicate 'Time' column
nok_combined = pd.concat([nok_data1, nok_data2.drop(columns=['Time'])], axis=1)

# Concatenate OK and NOK data along rows
data = pd.concat([ok_combined, nok_combined], ignore_index=True)
# Extract time-based features
data['Time'] = pd.to_datetime(data['Time'])
# data['Year'] = data['Time'].dt.year
# data['Month'] = data['Time'].dt.month
data['Hour'] = data['Time'].dt.hour
data['Minute'] = data['Time'].dt.minute
data['Second'] = data['Time'].dt.second
# data['DayOfWeek'] = data['Time'].dt.dayofweek
# # Drop the Time column
data = data.drop(columns=['Time'])

# Add labels
data['Label'] = 0  # No anomaly for OK data
data.loc[len(ok_combined):, 'Label'] = 1  # Anomaly for NOK data
# data = data.sample(frac=1).reset_index(drop=True)
# Separate features and target
X = data.drop(columns=['Label'])
y = data['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# print(X_train.sample())
# print("**************")
# print(X_test.sample())
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(report)
