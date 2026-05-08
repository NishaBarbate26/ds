# pip install pandas numpy matplotlib  # Install required libraries

import pandas as pd  # For dataset handling
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs

print("============================================")
print("PART 1: GROUPED DESCRIPTIVE STATISTICS")
print("============================================")

# Step 1: Create Dataset
data = {
    "Age": [23, 25, 31, 35, 45, 52, 23, 40, 60, 48],
    # Age values
    "Income": [25000, 27000, 32000, 40000, 50000, 52000, 26000, 42000, 60000, 48000],
    # Income values
}

df = pd.DataFrame(data)
# Converts dictionary into DataFrame

# Create Age Groups
bins = [20, 30, 40, 50, 70]
# Defines age intervals

labels = ["20-30", "30-40", "40-50", "50+"]
# Labels for age groups

df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels)
# Categorizes age values into groups

print("\nDataset:\n", df)
# Output: Displays dataset with Age_Group column

# Step 2: Grouped Statistics
grouped_stats = df.groupby("Age_Group", observed=True)["Income"].agg(
    Mean="mean",  # Average income
    Median="median",  # Middle value
    Minimum="min",  # Lowest income
    Maximum="max",  # Highest income
    Std_Dev="std",  # Standard deviation
)

print("\nGrouped Statistics:\n", grouped_stats)
# Output: Displays statistics for each age group

# Step 3: List per Category
income_lists = df.groupby("Age_Group", observed=True)["Income"].apply(list)
# Stores income values as list for each age group

print("\nIncome Lists per Age Group:\n", income_lists)
# Output: Displays list of incomes group-wise


# PART 2: IRIS DATASET

print("\n============================================")
print("PART 2: IRIS DATASET")
print("============================================")

# Load dataset safely
try:
    iris = pd.read_csv("Iris.csv")
    # Reads Iris dataset
except:
    print("Error: Iris.csv not found")
    # Output if file missing
    exit()

print("\nFirst 5 rows:\n", iris.head())
# Output: Displays first 5 rows of Iris dataset

# Drop Id if exists
if "Id" in iris.columns:
    iris = iris.drop(columns=["Id"])
    # Removes unnecessary Id column

# Overall Statistics
print("\nOverall Statistics:\n", iris.describe())
# Output: Displays statistical summary of all numerical columns

# Species-wise Statistics (ALL FEATURES)
species_stats = iris.groupby("Species").agg(["mean", "std", "min", "max", "median"])
# Calculates species-wise statistics

print("\nSpecies-wise Statistics (All Features):\n", species_stats)
# Output: Displays grouped statistics for each species

# Percentiles
print("\nPercentiles (25%, 50%, 75%):\n")

numeric_cols = iris.select_dtypes(include=["number"])
# Selects only numerical columns

print(numeric_cols.quantile([0.25, 0.5, 0.75]))
# Output: Displays quartiles/percentiles

# Species-wise Percentiles
species_percentiles = iris.groupby("Species").quantile([0.25, 0.5, 0.75])
# Calculates percentiles for each species

print("\nSpecies-wise Percentiles:\n", species_percentiles)
# Output: Displays species-wise percentile values

print("\n============================================")
print("PROGRAM COMPLETED SUCCESSFULLY ✅")
print("============================================")


# ============================================
# SIMPLE VISUALIZATIONS
# ============================================

print("\nGenerating Simple Plots...")

# 1. Histogram (Income Distribution)
plt.figure()

plt.hist(df["Income"], bins=5)
# Output: Shows frequency distribution of income values

plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()

# 2. Bar Chart (Mean Income per Age Group)
plt.figure()

grouped_stats["Mean"].plot(kind="bar")
# Output: Displays average income for each age group

plt.title("Mean Income per Age Group")
plt.xlabel("Age Group")
plt.ylabel("Mean Income")
plt.show()

# 3. Box Plot (Income Spread)
plt.figure()

plt.boxplot(df["Income"])
# Output: Shows spread, median, and outliers of income

plt.title("Income Box Plot")
plt.ylabel("Income")
plt.show()

# 4. Histogram (Iris Sepal Length)
plt.figure()

plt.hist(iris["SepalLengthCm"], bins=10)
# Output: Displays distribution of sepal length values

plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

# 5. Box Plot (Sepal Length by Species)
plt.figure()

iris.boxplot(column="SepalLengthCm", by="Species")
# Output: Compares sepal length distribution for each species

plt.title("Sepal Length by Species")

plt.suptitle("")
# Removes automatic extra title

plt.show()


# ============================================================
# COMPLETE OUTPUT EXPLANATION
# ============================================================

# 1. Dataset Output
# Displays Age, Income, and Age_Group columns.
# Age_Group categorizes people into ranges:
# 20-30, 30-40, 40-50, and 50+.

# 2. Grouped Statistics Output
# Displays:
# Mean -> average income
# Median -> middle income value
# Minimum -> lowest income
# Maximum -> highest income
# Std_Dev -> spread of income values

# Example:
# Age group 20-30 may have average income around 26000.

# 3. Income Lists Output
# Shows list of all incomes belonging to each age group.
# Example:
# 20-30 -> [25000, 27000, 26000]

# 4. Iris Dataset First 5 Rows Output
# Displays first 5 records of Iris dataset.
# Columns include:
# SepalLengthCm
# SepalWidthCm
# PetalLengthCm
# PetalWidthCm
# Species

# 5. Overall Statistics Output
# Shows statistical summary for all numerical columns:
# count, mean, std, min, max, quartiles.

# 6. Species-wise Statistics Output
# Calculates statistics separately for:
# Iris-setosa
# Iris-versicolor
# Iris-virginica

# Helps compare flower measurements species-wise.

# 7. Percentiles Output
# Shows:
# 25% percentile -> lower quartile
# 50% percentile -> median
# 75% percentile -> upper quartile

# Helps understand data distribution.

# 8. Species-wise Percentiles Output
# Displays percentile values for each flower species separately.

# 9. Histogram of Income
# Shows distribution of income frequencies.
# Taller bars indicate more people in that income range.

# 10. Bar Chart Output
# Displays mean income for each age group.
# Helps compare average income category-wise.

# 11. Income Box Plot Output
# Shows:
# Median income
# Spread of income values
# Possible outliers

# 12. Sepal Length Histogram Output
# Shows frequency distribution of SepalLengthCm values.

# 13. Sepal Length Box Plot Output
# Compares sepal length among flower species.
# Helps identify variation and outliers species-wise.

# Overall Aim:
# This program demonstrates:
# Grouped Descriptive Statistics
# Percentile Calculation
# Dataset Grouping
# Data Visualization
# and Statistical Analysis using
# custom dataset and Iris dataset.
