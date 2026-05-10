# pip install seaborn matplotlib
# Install required libraries:
# seaborn -> dataset loading and visualization
# matplotlib -> plotting graphs

import seaborn as sns  # For statistical visualization
import matplotlib.pyplot as plt  # For plotting graphs

# LOAD DATASET
df = sns.load_dataset("titanic")
# Loads Titanic dataset from seaborn library

print("First 5 Rows:\n", df.head())
# Output: Displays first 5 rows of Titanic dataset

print("\nMissing Values:\n", df.isnull().sum())
# Output: Displays total missing values in each column


# BOX PLOT
plt.figure()

sns.boxplot(
    x="sex",  # Gender on X-axis
    y="age",  # Age on Y-axis
    hue="survived",  # Separate colors based on survival
    data=df,
)
# Creates boxplot comparing age distribution
# by gender and survival status

plt.title("Age Distribution by Gender and Survival")

plt.xlabel("Gender")
plt.ylabel("Age")

plt.show()
# Displays boxplot graph


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

# Example:
# survived = 0 -> passenger did not survive
# survived = 1 -> passenger survived

# 2. Missing Values Output
# Displays total null/missing values in each column.

# Example:
# age column may contain missing values.
# cabin column usually contains many missing values.

# Helps identify incomplete data.

# 3. Box Plot Output
# Boxplot visualizes age distribution based on:
# Gender (male/female)
# Survival status (survived/not survived)

# x-axis:
# male and female categories

# y-axis:
# passenger age values

# hue='survived':
# Different colors represent:
# 0 -> not survived
# 1 -> survived

# 4. Understanding Box Plot Components

# Middle Line:
# Represents median age

# Box:
# Represents middle 50% data (Interquartile Range)

# Upper and Lower Lines (Whiskers):
# Show spread of normal data

# Dots outside whiskers:
# Represent outliers/extreme age values

# 5. Observations from Plot
# Helps compare:
# Age distribution of males vs females
# Survival patterns among age groups

# Example observations:
# Younger passengers may have higher survival rate.
# Female passengers may show better survival distribution.

# Overall Aim:
# This program demonstrates:
# Data visualization using boxplot
# Missing value analysis
# Age distribution comparison
# Gender-wise survival analysis
# and pattern identification in Titanic dataset.
