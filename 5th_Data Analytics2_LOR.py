# pip install pandas numpy matplotlib seaborn scikit-learn
# Install required libraries:
# pandas -> dataset handling
# numpy -> numerical operations
# matplotlib & seaborn -> visualization
# scikit-learn -> machine learning algorithms

import numpy as np  # For numerical calculations
import pandas as pd  # For handling dataset
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For advanced visualizations

from sklearn.model_selection import train_test_split

# Splits dataset into training and testing sets

from sklearn.preprocessing import StandardScaler

# Used for feature scaling/standardization

from sklearn.linear_model import LogisticRegression

# Imports Logistic Regression model

from sklearn.metrics import confusion_matrix, accuracy_score

# Metrics for model evaluation


# STEP 1: LOAD DATASET
df = pd.read_csv("Social_Network_Ads.csv")
# Reads CSV dataset into DataFrame

print("First 5 Rows:\n", df.head())
# Output: Displays first 5 records of dataset

print("\nDataset Info:\n")
print(df.info())
# Output: Shows rows, columns, datatypes, non-null values

print("\nStatistical Summary:\n", df.describe())
# Output: Displays mean, std, min, max, quartiles etc.

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())
# Output: Shows missing values count in each column


# STEP 2: DATA PREPROCESSING

# Selecting relevant features
X = df[["Age", "EstimatedSalary"]]
# Input features used for prediction

y = df["Purchased"]
# Target variable/output
# 0 = Not Purchased
# 1 = Purchased

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
# Splits dataset:
# 75% -> training data
# 25% -> testing data

# Feature Scaling
scaler = StandardScaler()
# Creates scaler object

X_train = scaler.fit_transform(X_train)
# Fits and transforms training data

X_test = scaler.transform(X_test)
# Transforms test data using same scaling


# STEP 3: MODEL BUILDING

model = LogisticRegression()
# Creates Logistic Regression model

model.fit(X_train, y_train)
# Trains model using training data


# STEP 4: PREDICTIONS
y_pred = model.predict(X_test)
# Predicts purchase values for test data

print("\nPredictions:\n", y_pred)
# Output: Displays predicted class values (0 or 1)


# STEP 5: CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
# Creates confusion matrix

print("\nConfusion Matrix:\n", cm)
# Output: Displays matrix of actual vs predicted values

TN, FP, FN, TP = cm.ravel()
# Extracts:
# TN -> True Negative
# FP -> False Positive
# FN -> False Negative
# TP -> True Positive

print("\nTN:", TN)
print("FP:", FP)
print("FN:", FN)
print("TP:", TP)

# STEP 6: PERFORMANCE METRICS
accuracy = accuracy_score(y_test, y_pred)
# Calculates prediction accuracy

error_rate = 1 - accuracy
# Calculates error rate

precision = TP / (TP + FP) if (TP + FP) != 0 else 0
# Precision formula

recall = TP / (TP + FN) if (TP + FN) != 0 else 0
# Recall formula

print("\nAccuracy:", accuracy)
# Output: Displays model accuracy

print("Error Rate:", error_rate)
# Output: Displays error percentage

print("Precision:", precision)
# Output: Displays precision value

print("Recall:", recall)
# Output: Displays recall value


# STEP 7: VISUALIZATION
plt.figure()

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# Heatmap visualization of confusion matrix

plt.title("Confusion Matrix Heatmap")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
# Displays heatmap graph


# ============================================================
# COMPLETE OUTPUT EXPLANATION
# ============================================================

# 1. First 5 Rows Output
# Displays first 5 records of Social Network Ads dataset.
# Columns may include:
# User ID
# Gender
# Age
# EstimatedSalary
# Purchased

# 2. Dataset Info Output
# Displays:
# Number of rows and columns
# Datatypes
# Non-null values
# Memory usage

# 3. Statistical Summary Output
# Displays:
# Mean -> average value
# Std -> standard deviation
# Min -> minimum value
# Max -> maximum value
# Quartiles -> data distribution

# 4. Missing Values Output
# Shows number of null values in each column.
# If values are 0, dataset has no missing values.

# 5. Feature Selection Output
# X contains:
# Age
# EstimatedSalary

# y contains:
# Purchased column
# 0 -> customer did not purchase
# 1 -> customer purchased

# 6. Train-Test Split Output
# Dataset divided into:
# Training data -> used for learning
# Testing data -> used for evaluation

# 7. Feature Scaling Output
# StandardScaler standardizes data.
# Converts values into similar scale.
# Helps Logistic Regression perform better.

# Formula:
# z = (x - mean) / standard deviation

# 8. Model Training Output
# Logistic Regression learns relationship between:
# Age + Salary and Purchase decision.

# 9. Predictions Output
# Displays predicted class values:
# 0 -> Not Purchased
# 1 -> Purchased

# Example:
# [0 1 0 1 1]

# 10. Confusion Matrix Output
# Example Matrix:
# [[TN FP]
#  [FN TP]]

# TN -> Correctly predicted not purchased
# FP -> Incorrectly predicted purchased
# FN -> Incorrectly predicted not purchased
# TP -> Correctly predicted purchased

# 11. Accuracy Output
# Measures overall prediction correctness.

# Formula:
# Accuracy = (TP + TN) / Total Predictions

# Higher accuracy means better model performance.

# 12. Error Rate Output
# Measures incorrect predictions.

# Formula:
# Error Rate = 1 - Accuracy

# Lower error rate means better model.

# 13. Precision Output
# Measures how many predicted positives are actually correct.

# Formula:
# Precision = TP / (TP + FP)

# High precision means fewer false positives.

# 14. Recall Output
# Measures how many actual positives are correctly predicted.

# Formula:
# Recall = TP / (TP + FN)

# High recall means fewer false negatives.

# 15. Confusion Matrix Heatmap Output
# Visual representation of confusion matrix.
# Darker boxes represent higher values.
# Helps understand classification performance visually.

# Overall Aim:
# This program demonstrates:
# Logistic Regression Classification
# Data preprocessing
# Feature scaling
# Customer purchase prediction
# Confusion Matrix analysis
# Accuracy, Precision, Recall calculation
# and visualization using heatmap.
