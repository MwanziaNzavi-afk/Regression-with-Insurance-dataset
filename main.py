#%%
# Load Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb

# Set File Paths
train_path = "C:/Users/mwanz/OneDrive/Desktop/Regression with an Insurance Dataset/train.csv"
test_path = "C:/Users/mwanz/OneDrive/Desktop/Regression with an Insurance Dataset/test.csv"
submission_path = "C:/Users/mwanz/OneDrive/Desktop/Regression with an Insurance Dataset/sample_submission.csv"

# Load Datasets
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
sample_submission = pd.read_csv(submission_path)

# Data Exploration
print("Structure of Train Data:")
print(train_data.info())

print("Structure of Test Data:")
print(test_data.info())

# Summary of the target variable
print(train_data["Premium Amount"].describe())

# Check for missing values
print("Missing values in Train Data:")
print(train_data.isna().sum())

print("Missing values in Test Data:")
print(test_data.isna().sum())

# Visualize the Target Variable
sns.histplot(train_data["Premium Amount"], bins=30, kde=False, color="blue")
plt.title("Distribution of Premium Amount")
plt.xlabel("Premium Amount")
plt.ylabel("Frequency")
plt.show()

# Log-transform the target variable
train_data["LogPremium"] = np.log1p(train_data["Premium Amount"])

# Remove irrelevant columns (e.g., id)
if "id" in train_data.columns:
    train_data = train_data.drop(columns=["id"])
if "id" in test_data.columns:
    test_data = test_data.drop(columns=["id"])

# Handle missing values (replace with mean for simplicity)
train_data = train_data.fillna(train_data.mean())
test_data = test_data.fillna(test_data.mean())

# Normalize numeric features
numeric_features = train_data.select_dtypes(include=[np.number]).columns
train_data[numeric_features] = (train_data[numeric_features] - train_data[numeric_features].mean()) / train_data[numeric_features].std()
test_data[numeric_features] = (test_data[numeric_features] - test_data[numeric_features].mean()) / test_data[numeric_features].std()

# Split train data into training and validation sets
X = train_data.drop(columns=["LogPremium", "Premium Amount"])
y = train_data["LogPremium"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=123)

# Define RMSLE function
def rmsle(pred, actual):
    return np.sqrt(mean_squared_log_error(actual, pred))

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, max_features=5, random_state=123)
rf_model.fit(X_train, y_train)
rf_predictions = np.expm1(rf_model.predict(X_valid))
rf_rmsle = rmsle(rf_predictions, np.expm1(y_valid))

# Train XGBoost Model
xgb_model = xgb.XGBRegressor(n_estimators=100, objective="reg:squarederror", random_state=123)
xgb_model.fit(X_train, y_train)
xgb_predictions = np.expm1(xgb_model.predict(X_valid))
xgb_rmsle = rmsle(xgb_predictions, np.expm1(y_valid))

# Compare RMSLE
print("Random Forest RMSLE:", rf_rmsle)
print("XGBoost RMSLE:", xgb_rmsle)

# Generate predictions on the test set using the best model (e.g., Random Forest)
test_predictions = np.expm1(rf_model.predict(test_data))

# Create submission file
submission = sample_submission.copy()
submission["Premium Amount"] = test_predictions
submission.to_csv("submission.csv", index=False)

print("Submission file created successfully as 'submission.csv'.")
#%%
