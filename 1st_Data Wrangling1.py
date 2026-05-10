# pip install pandas numpy matplotlib seaborn  # Install required libraries

import pandas as pd  # For dataset handling
import numpy as np  # For numerical calculations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For advanced visualization

print("Libraries imported successfully!\n")  # Output: Libraries loaded successfully


# DATASET SOURCE
# Titanic Dataset from Kaggle

print("Dataset: Titanic Dataset")
print("Source: Kaggle\n")


# LOAD DATASET
file_path = "Titanic-Dataset.csv"  # Dataset file path
df = pd.read_csv(file_path)  # Reads CSV file into DataFrame

print("\nDataset Loaded Successfully!\n")  # Output: Dataset loaded successfully


# BASIC DATA EXPLORATION

print("First 5 Rows of Dataset:")
print(df.head())  # Output: Displays first 5 rows of Titanic dataset

print("\nColumns in Dataset:")
print(df.columns)  # Output: Shows all dataset column names

print("\nShape of Dataset (Rows, Columns):")
print(df.shape)  # Output example: (891,12) -> 891 rows and 12 columns


# VARIABLE DESCRIPTION
print("\nVariable Description:")
print("Survived: 0 = No, 1 = Yes")
print("Pclass: Passenger class (1, 2, 3)")
print("Sex: Gender")
print("Age: Age of passenger")
print("Fare: Ticket price")
print("Embarked: Port of embarkation\n")


# DATA PREPROCESSING
print("Missing Values in Each Column:")
print(df.isnull().sum())  # Output: Shows missing/null values in each column

print("\nStatistical Summary:")
print(df.describe())  # Output: Shows mean, min, max, std deviation etc.

print("\nData Types of Each Column:")
print(df.dtypes)  # Output: Displays datatype of every column

print("\nData Type Explanation:")
print("int64 -> Integer values")
print("float64 -> Decimal values")
print("object -> Text/Categorical values\n")


# DATA CLEANING
print("----- Data Cleaning -----")

df["Age"] = df["Age"].fillna(df["Age"].mean())
# Missing Age values replaced with average age

print("Missing values in Age filled with MEAN")

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
# Missing Embarked values replaced using most frequent value

print("Missing values in Embarked filled with MODE")

if "Cabin" in df.columns:
    df = df.drop(columns=["Cabin"])
    # Removes Cabin column due to many missing values

    print("Cabin column dropped")


# DATA NORMALIZATION
print("\n----- Data Normalization -----")

df["Fare"] = (df["Fare"] - df["Fare"].min()) / (df["Fare"].max() - df["Fare"].min())
# Converts Fare values between 0 and 1 using Min-Max Scaling

print("Fare column normalized using Min-Max Scaling")


# DATA TYPE CONVERSION
print("\n----- Data Type Conversion -----")

df["Age"] = df["Age"].astype(float)
# Converts Age datatype into float

print("Age converted to float")


# CATEGORICAL TO NUMERICAL

print("\n----- Converting Categorical to Numerical -----")

if "Sex" in df.columns:
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    # Converts male=0 and female=1

    print("Sex encoded")

if "Embarked" in df.columns:
    df = pd.get_dummies(df, columns=["Embarked"])
    # One-Hot Encoding creates separate columns for categories

    print("Embarked converted using One-Hot Encoding")


# FINAL OUTPUT

print("\nProcessed Dataset (First 5 Rows):")
print(df.head())
# Output: Shows processed and cleaned first 5 rows

print("\nFinal Dataset Info:")
print(df.info())
# Output: Displays rows, columns, datatypes, non-null values

print("\nData Wrangling Completed Successfully!")


# DATA VISUALIZATION

print("\n----- Data Visualization -----")


# 1. Histogram of Age
plt.figure()
plt.hist(df["Age"], bins=30)
# Output: Shows distribution of passenger ages

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# 2. Histogram of Fare (Normalized)
plt.figure()
plt.hist(df["Fare"], bins=30)
# Output: Shows normalized Fare distribution

plt.title("Normalized Fare Distribution")
plt.xlabel("Fare (0–1)")
plt.ylabel("Frequency")
plt.show()


# 3. Count Plot for Survival
plt.figure()
sns.countplot(x="Survived", data=df)
# Output: Shows count of survived vs non-survived passengers

plt.title("Survival Count (0 = No, 1 = Yes)")
plt.show()


# 4. Boxplot for Age
plt.figure()
sns.boxplot(y=df["Age"])
# Output: Shows median, spread and outliers in Age column

plt.title("Boxplot of Age")
plt.show()


# 5. Correlation Heatmap
plt.figure()
sns.heatmap(df.select_dtypes(include=["number"]).corr(), annot=True)
# Output: Shows correlation between numerical columns

plt.title("Correlation Heatmap")
plt.show()


# ============================================================
# COMPLETE OUTPUT EXPLANATION
# ============================================================

# 1. Libraries imported successfully
# Meaning:
# All required libraries are loaded properly without errors.

# 2. Dataset Loaded Successfully
# Meaning:
# Titanic CSV dataset is successfully read into DataFrame.

# 3. df.head() Output
# Displays first 5 rows of dataset.
# Helps understand sample records and dataset structure.

# 4. df.columns Output
# Shows all column names such as:
# PassengerId, Survived, Pclass, Name, Sex, Age, Fare, Embarked etc.

# 5. df.shape Output
# Example Output: (891,12)
# Meaning:
# 891 rows = passengers
# 12 columns = features/attributes

# 6. Missing Values Output
# Shows number of null values in each column.
# Example:
# Age may contain 177 missing values.
# Cabin may contain many missing values.

# 7. Statistical Summary Output
# Shows:
# Mean -> average value
# Min -> minimum value
# Max -> maximum value
# Std -> standard deviation
# Helps understand numerical data distribution.

# 8. Data Cleaning Output
# Missing Age values replaced using mean.
# Missing Embarked values replaced using mode.
# Cabin column removed due to excessive null values.

# 9. Fare Normalization Output
# Fare values converted between 0 and 1.
# Helps improve machine learning performance.

# 10. Encoding Output
# Sex column:
# male -> 0
# female -> 1

# Embarked column:
# Converted into separate binary columns using One-Hot Encoding.

# 11. Final Dataset Info Output
# Shows:
# Total rows and columns
# Non-null values
# Datatypes
# Memory usage
# Confirms dataset preprocessing completed successfully.

# 12. Histogram of Age
# Graph shows distribution of passenger ages.
# Most passengers are adults between 20–40 years.

# 13. Histogram of Fare
# Shows normalized fare distribution.
# Most passengers paid lower ticket fares.

# 14. Survival Count Plot
# Compares survived and non-survived passengers.
# Usually more passengers died than survived.

# 15. Boxplot of Age
# Shows:
# Median age
# Spread of age values
# Outliers/extreme ages

# 16. Correlation Heatmap
# Displays relationships between numerical columns.
# Correlation values:
# +1 -> strong positive relation
# -1 -> strong negative relation
# 0 -> no relation

# Overall Aim:
# This program performs Data Wrangling and Data Visualization
# on Titanic dataset to clean, preprocess, and analyze data
# before applying Machine Learning algorithms.
