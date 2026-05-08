# pip install pandas numpy matplotlib seaborn scikit-learn
# Install required libraries:
# pandas -> dataset handling
# numpy -> numerical operations
# matplotlib & seaborn -> visualization
# scikit-learn -> machine learning algorithms

import numpy as np  # For numerical calculations
import pandas as pd  # For handling datasets
import seaborn as sns  # For advanced visualizations
import matplotlib.pyplot as plt  # For plotting graphs

from sklearn.model_selection import train_test_split

# Splits dataset into training and testing sets

from sklearn.naive_bayes import GaussianNB

# Imports Gaussian Naive Bayes classifier

from sklearn.metrics import confusion_matrix, accuracy_score

# Metrics for evaluating model performance


# STEP 1: LOAD DATASET
df = pd.read_csv("Iris.csv")
# Reads Iris dataset from CSV file

print("First 5 Rows:\n", df.head())
# Output: Displays first 5 rows of Iris dataset

print("\nDataset Info:\n")
print(df.info())
# Output: Shows rows, columns, datatypes, non-null values

print("\nStatistical Summary:\n")
print(df.describe())
# Output: Displays mean, std, min, max, quartiles etc.

print("\nMissing Values:\n", df.isnull().sum())
# Output: Shows number of missing values in each column

# Drop Id column if exists
if "Id" in df.columns:
    df = df.drop(columns=["Id"])
    # Removes unnecessary Id column


# STEP 2: FEATURES & TARGET

X = df.drop(columns=["Species"])
# Input features:
# SepalLengthCm
# SepalWidthCm
# PetalLengthCm
# PetalWidthCm

y = df["Species"]
# Target/output variable:
# Flower species names


# STEP 3: TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
# Splits dataset:
# 75% -> training data
# 25% -> testing data


# STEP 4: MODEL TRAINING
model = GaussianNB()
# Creates Gaussian Naive Bayes model

model.fit(X_train, y_train)
# Trains model using training data


# STEP 5: PREDICTION
y_pred = model.predict(X_test)
# Predicts flower species using test data

print("\nPredicted Values:\n", y_pred)
# Output: Displays predicted species names


# STEP 6: CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
# Creates confusion matrix

print("\nConfusion Matrix:\n", cm)
# Output: Displays classification results matrix

# Overall Accuracy
overall_accuracy = accuracy_score(y_test, y_pred)
# Calculates overall model accuracy

print("\nOverall Accuracy:", overall_accuracy)
# Output: Displays overall accuracy score


# STEP 7: METRICS PER CLASS
classes = np.unique(y)
# Gets unique flower species names

for i, cls in enumerate(classes):

    TP = cm[i][i]
    # True Positives

    FP = sum(cm[:, i]) - TP
    # False Positives

    FN = sum(cm[i, :]) - TP
    # False Negatives

    TN = cm.sum() - (TP + FP + FN)
    # True Negatives

    accuracy = (TP + TN) / cm.sum()
    # Accuracy formula

    error_rate = 1 - accuracy
    # Error rate formula

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    # Precision formula

    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    # Recall formula

    print(f"\nClass: {cls}")

    print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)
    print("TN:", TN)

    print("Accuracy:", accuracy)
    print("Error Rate:", error_rate)
    print("Precision:", precision)
    print("Recall:", recall)


# STEP 8: VISUALIZATION

plt.figure()

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
)
# Heatmap visualization of confusion matrix

plt.title("Confusion Matrix (Naïve Bayes - Iris)")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
# Displays confusion matrix heatmap


# ============================================================
# COMPLETE OUTPUT EXPLANATION
# ============================================================

# 1. First 5 Rows Output
# Displays first 5 records of Iris dataset.
# Columns include:
# SepalLengthCm
# SepalWidthCm
# PetalLengthCm
# PetalWidthCm
# Species

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
# Shows total null values in each column.
# If output values are 0, dataset has no missing values.

# 5. Feature and Target Output
# X contains flower measurements/features.
# y contains flower species labels:
# Iris-setosa
# Iris-versicolor
# Iris-virginica

# 6. Train-Test Split Output
# Dataset divided into:
# Training Data -> used for learning
# Testing Data -> used for evaluation

# 7. Model Training Output
# Gaussian Naive Bayes model learns probability patterns
# between flower measurements and species.

# Naive Bayes works on probability concept
# using Bayes Theorem.

# 8. Predicted Values Output
# Displays predicted flower species names.
# Example:
# ['Iris-setosa', 'Iris-versicolor', ...]

# 9. Confusion Matrix Output
# Matrix compares actual vs predicted classes.

# Example:
# [[TP  0  0]
#  [ 0 TP  1]
#  [ 0  2 TP]]

# Diagonal values = correct predictions
# Off-diagonal values = wrong predictions

# 10. Overall Accuracy Output
# Measures total prediction correctness.

# Formula:
# Accuracy = Correct Predictions / Total Predictions

# Higher accuracy means better classification performance.

# 11. TP, FP, FN, TN Output
# TP -> Correctly predicted class
# FP -> Incorrectly predicted as class
# FN -> Incorrectly missed class
# TN -> Correctly predicted not belonging to class

# 12. Precision Output
# Measures correctness of positive predictions.

# Formula:
# Precision = TP / (TP + FP)

# High precision means fewer false positives.

# 13. Recall Output
# Measures ability to identify actual positives.

# Formula:
# Recall = TP / (TP + FN)

# High recall means fewer false negatives.

# 14. Error Rate Output
# Measures incorrect predictions.

# Formula:
# Error Rate = 1 - Accuracy

# Lower error rate means better model.

# 15. Heatmap Output
# Visual representation of confusion matrix.
# Darker boxes represent higher values.
# Helps understand classification performance visually.

# Overall Aim:
# This program demonstrates:
# Gaussian Naive Bayes Classification
# Data preprocessing
# Iris flower species prediction
# Confusion Matrix analysis
# Accuracy, Precision, Recall calculation
# and visualization using heatmap.
