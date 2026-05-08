# pip install pandas matplotlib scikit-learn
# Install required libraries:
# pandas -> dataset handling
# matplotlib -> plotting graphs
# scikit-learn -> machine learning algorithms

import pandas as pd  # For handling dataset using DataFrame
import matplotlib.pyplot as plt  # For visualization

from sklearn.model_selection import train_test_split

# Splits dataset into training and testing sets

from sklearn.linear_model import LinearRegression

# Imports Linear Regression model

from sklearn.metrics import mean_squared_error, r2_score

# Metrics for evaluating model performance


# STEP 1: LOAD DATASET
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
# Dataset URL

df = pd.read_csv(url)
# Reads dataset from URL into DataFrame

print("\nFirst 5 Rows:\n", df.head())
# Output: Displays first 5 rows of Boston Housing dataset


# STEP 2: DATA UNDERSTANDING
print("\nDataset Shape:", df.shape)
# Output example: (506,14)
# 506 rows and 14 columns

print("\nColumn Names:\n", df.columns)
# Output: Displays all column names/features

print("\nData Info:\n")
print(df.info())
# Output: Shows datatype, non-null values, memory usage

print("\nStatistical Summary:\n", df.describe())
# Output: Displays mean, std, min, max, quartiles etc.

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())
# Output: Displays missing values count in each column


# STEP 3: FEATURES & TARGET

X = df.drop("medv", axis=1)
# Features/Input variables
# Drops 'medv' column and stores remaining columns in X

y = df["medv"]
# Target variable/output
# medv = median value of house prices


# STEP 4: TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Splits dataset:
# 80% -> training data
# 20% -> testing data
# random_state=42 ensures same random split every run


# STEP 5: MODEL TRAINING
model = LinearRegression()
# Creates Linear Regression model object

model.fit(X_train, y_train)
# Trains model using training data


# STEP 6: PREDICTION
y_pred = model.predict(X_test)
# Predicts house prices using test data

print("\nPredicted Prices:\n", y_pred[:5])
# Output: Displays first 5 predicted house prices


# STEP 7: MODEL EVALUATION
mse = mean_squared_error(y_test, y_pred)
# Calculates Mean Squared Error

r2 = r2_score(y_test, y_pred)
# Calculates R2 Score

print("\nMean Squared Error:", mse)
# Output: Displays prediction error value

print("R2 Score:", r2)
# Output: Displays model accuracy/performance score


# STEP 8: VISUALIZATION
plt.figure()

plt.scatter(y_test, y_pred)
# Scatter plot comparing actual vs predicted house prices

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

plt.show()
# Displays graph


# ============================================================
# COMPLETE OUTPUT EXPLANATION
# ============================================================

# 1. First 5 Rows Output
# Displays first 5 records of Boston Housing dataset.
# Columns represent housing features like:
# crime rate
# number of rooms
# tax rate
# house price etc.

# 2. Dataset Shape Output
# Example Output: (506,14)
# Meaning:
# 506 rows -> total records
# 14 columns -> total features including target variable

# 3. Column Names Output
# Displays all feature names in dataset.
# Example:
# crim, rm, age, tax, ptratio, medv etc.

# 4. Data Info Output
# Displays:
# Datatypes
# Non-null values
# Memory usage
# Helps understand dataset structure.

# 5. Statistical Summary Output
# Displays:
# Mean -> average value
# Std -> standard deviation
# Min -> minimum value
# Max -> maximum value
# Quartiles -> data distribution

# 6. Missing Values Output
# Shows total missing values in each column.
# If output is 0, dataset has no missing values.

# 7. Feature and Target Output
# X contains input features.
# y contains target variable 'medv'
# medv = house price value.

# 8. Train-Test Split Output
# Dataset divided into:
# Training Data -> used for learning model
# Testing Data -> used for evaluating model

# 9. Model Training Output
# Linear Regression model learns relationship between:
# housing features and house prices.

# 10. Predicted Prices Output
# Displays first 5 predicted house prices.
# Example:
# [28.5, 31.2, 19.8 ...]
# These are predicted by model using test data.

# 11. Mean Squared Error Output
# Measures average squared prediction error.
# Lower MSE means better model performance.

# Formula:
# MSE = average of (actual - predicted)^2

# 12. R2 Score Output
# Measures accuracy of model.
# Range:
# 0 to 1

# Closer to 1 -> better prediction accuracy
# Example:
# R2 = 0.75 means model explains 75% variance.

# 13. Scatter Plot Output
# Compares actual prices with predicted prices.
# If points are close to straight diagonal pattern,
# model predictions are good.

# Overall Aim:
# This program demonstrates:
# Linear Regression
# Data preprocessing
# Train-test splitting
# House price prediction
# Model evaluation using MSE and R2 score
# and visualization of prediction results.
