# pip install seaborn matplotlib pandas
# Install required libraries:
# seaborn -> dataset loading and visualization
# matplotlib -> plotting graphs
# pandas -> dataset handling

import seaborn as sns  # For statistical visualization
import matplotlib.pyplot as plt  # For plotting graphs
import pandas as pd  # For handling datasets

# STEP 1: LOAD DATASET
df = sns.load_dataset("titanic")
# Loads Titanic dataset from seaborn library

print("First 5 Rows:\n", df.head())
# Output: Displays first 5 rows of Titanic dataset

print("\nDataset Info:\n")
print(df.info())
# Output: Displays:
# number of rows/columns
# datatypes
# non-null values
# memory usage

print("\nStatistical Summary:\n")
print(df.describe())
# Output: Displays statistical summary:
# mean, std, min, max, quartiles etc.


# STEP 2: FIND PATTERNS IN DATA

# 1. Survival Count
plt.figure()

sns.countplot(x="survived", data=df)
# Count plot showing number of survived and non-survived passengers

plt.title("Survival Count (0 = No, 1 = Yes)")

plt.show()
# Displays graph


# 2. Survival based on Gender
plt.figure()

sns.countplot(x="sex", hue="survived", data=df)
# Count plot comparing survival based on gender

plt.title("Survival by Gender")

plt.show()
# Displays graph


# 3. Survival based on Passenger Class
plt.figure()

sns.countplot(x="pclass", hue="survived", data=df)
# Count plot comparing survival based on passenger class

plt.title("Survival by Passenger Class")

plt.show()
# Displays graph


# 4. Correlation Heatmap
plt.figure()

numeric_df = df.select_dtypes(include=["number"])
# Selects only numerical columns

sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
# Creates heatmap showing correlation between numerical features

plt.title("Correlation Heatmap")

plt.show()
# Displays heatmap


# STEP 3: HISTOGRAM OF FARE

plt.figure()

sns.histplot(df["fare"], bins=30, kde=True)
# Histogram showing distribution of ticket fare
# kde=True adds smooth density curve

plt.title("Distribution of Ticket Fare")

plt.xlabel("Fare")
plt.ylabel("Frequency")

plt.show()
# Displays histogram


# ============================================================
# COMPLETE OUTPUT EXPLANATION
# ============================================================

# 1. First 5 Rows Output
# Displays first 5 records of Titanic dataset.

# Columns may include:
# survived
# pclass
# sex
# age
# fare
# embarked etc.

# 2. Dataset Info Output
# Displays:
# Total rows and columns
# Datatypes of columns
# Non-null values
# Memory usage

# Example:
# float64 -> decimal values
# int64 -> integer values
# object -> text/categorical values

# 3. Statistical Summary Output
# Displays:
# Mean -> average value
# Std -> standard deviation
# Min -> minimum value
# Max -> maximum value
# Quartiles -> data distribution

# Helps understand numerical data characteristics.

# 4. Survival Count Plot Output
# Displays number of passengers:
# 0 -> did not survive
# 1 -> survived

# Taller bar means more passengers in that category.

# Helps understand overall survival distribution.

# 5. Survival by Gender Plot Output
# Compares male and female survival counts.

# hue='survived' separates:
# survived passengers
# non-survived passengers

# Observation:
# Females usually had higher survival rate than males.

# 6. Survival by Passenger Class Plot Output
# Compares survival among:
# 1st class
# 2nd class
# 3rd class passengers

# Observation:
# Higher-class passengers had better survival chances.

# 7. Correlation Heatmap Output
# Displays relationship between numerical columns.

# Correlation range:
# +1 -> strong positive relation
# -1 -> strong negative relation
# 0 -> no relation

# Darker colors indicate stronger correlation.

# Example:
# Fare may positively correlate with passenger class.

# 8. Fare Histogram Output
# Displays distribution of ticket fares.

# Histogram bars show frequency of fare ranges.
# KDE curve shows smooth probability distribution.

# Observation:
# Most passengers paid lower fares,
# while few passengers paid very high fares.

# Overall Aim:
# This program demonstrates:
# Exploratory Data Analysis (EDA)
# Pattern identification in Titanic dataset
# Count plots
# Correlation analysis
# Histogram visualization
# and understanding passenger survival patterns.
