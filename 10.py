# pip install pandas matplotlib seaborn
# Install required libraries:
# pandas -> dataset handling
# matplotlib -> plotting graphs
# seaborn -> advanced visualization

import pandas as pd  # For handling datasets
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For statistical visualization

# LOAD DATASET
df = pd.read_csv("Iris.csv")
# Reads Iris dataset from CSV file

print("\nFirst 5 Rows:\n", df.head())
# Output: Displays first 5 rows of Iris dataset


# Drop Id if exists
if "Id" in df.columns:
    df = df.drop(columns=["Id"])
    # Removes unnecessary Id column


# Rename columns
if len(df.columns) == 5:
    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]
    # Renames columns into simple readable names

print("\nDataset Info:\n")
print(df.info())
# Output: Displays:
# total rows and columns
# datatypes
# non-null values
# memory usage

print("\nFeature Types:\n")
print(df.dtypes)
# Output: Displays datatype of each column


# HISTOGRAMS
df.hist(figsize=(10, 8))
# Creates histograms for all numerical features

plt.suptitle("Histogram of Iris Features")

plt.show()
# Displays histogram graphs


# BOXPLOTS
plt.figure(figsize=(10, 6))

sns.boxplot(data=df.drop("species", axis=1))
# Creates boxplot for numerical features
# species column removed because it is categorical

plt.title("Boxplot of Iris Features")

plt.show()
# Displays boxplot


# Boxplot by species
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
# List of numerical features

for feature in features:

    plt.figure()

    sns.boxplot(x="species", y=feature, data=df)
    # Creates boxplot comparing each feature with species

    plt.title(f"{feature} vs Species")

    plt.show()
    # Displays boxplot graph


# ============================================================
# COMPLETE OUTPUT EXPLANATION
# ============================================================

# 1. First 5 Rows Output
# Displays first 5 records of Iris dataset.

# Columns include:
# sepal_length
# sepal_width
# petal_length
# petal_width
# species

# Species examples:
# Iris-setosa
# Iris-versicolor
# Iris-virginica

# 2. Drop Id Column Output
# Removes unnecessary Id column if present.
# Helps focus only on useful data features.

# 3. Rename Columns Output
# Renames dataset columns into simple readable names.
# Makes analysis and visualization easier.

# 4. Dataset Info Output
# Displays:
# Number of rows and columns
# Datatypes
# Non-null values
# Memory usage

# Example:
# float64 -> decimal values
# object -> text/categorical values

# 5. Feature Types Output
# Displays datatype of each feature.

# Example:
# sepal_length -> float64
# species -> object

# Helps identify numerical and categorical columns.

# 6. Histogram Output
# Histograms display frequency distribution of features.

# Features visualized:
# sepal_length
# sepal_width
# petal_length
# petal_width

# Taller bars indicate more frequent values.

# Helps understand:
# data spread
# skewness
# distribution pattern

# 7. Boxplot Output
# Displays spread of numerical features.

# Components of boxplot:
# Middle line -> median
# Box -> interquartile range (IQR)
# Whiskers -> data spread
# Dots outside -> outliers

# Helps detect:
# outliers
# variation
# feature distribution

# 8. Species-wise Boxplot Output
# Compares each feature across flower species.

# Example:
# petal_length vs species
# sepal_width vs species

# Helps identify differences among:
# Iris-setosa
# Iris-versicolor
# Iris-virginica

# Example observation:
# Iris-setosa usually has smaller petal length.
# Iris-virginica usually has larger petal dimensions.

# Overall Aim:
# This program demonstrates:
# Exploratory Data Analysis (EDA)
# Dataset preprocessing
# Histograms
# Boxplots
# Feature distribution analysis
# Species-wise comparison
# and visualization of Iris dataset features.
