# pip install pandas numpy matplotlib seaborn  # Install required libraries

import pandas as pd  # For handling tabular data using DataFrame
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For advanced statistical visualization

# STEP 1: CREATE DATASET
data = {
    "Student_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Unique student IDs
    "Name": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    # Student names
    "Maths": [85, 90, np.nan, 35, 95, 88, 76, 120, 67, 54],
    # Maths marks (contains missing value and inconsistency)
    "Science": [78, 85, 88, np.nan, 92, 84, 79, 300, 70, 60],
    # Science marks (contains missing value and extreme value)
    "English": [80, 82, 85, 70, np.nan, 86, 78, 90, 65, 55],
    # English marks (contains missing value)
    "Attendance (%)": [90, 95, 85, 60, 98, np.nan, 88, 110, 75, 65],
    # Attendance percentage (contains missing value and invalid value >100)
}

df = pd.DataFrame(data)
# Converts dictionary into DataFrame/table format

print("\nOriginal Dataset:\n", df)
# Output: Displays original dataset with missing values and inconsistencies


# STEP 2: HANDLE MISSING VALUES
print("\nMissing Values:\n", df.isnull().sum())
# Output: Shows number of null values in each column

# Using mean imputation
for col in ["Maths", "Science", "English", "Attendance (%)"]:
    df[col] = df[col].fillna(df[col].mean())
    # Replaces missing values with column mean/average

# HANDLE INCONSISTENCIES
# Valid range: 0 to 100
for col in ["Maths", "Science", "English", "Attendance (%)"]:

    df[col] = np.where(df[col] > 100, 100, df[col])
    # Values greater than 100 replaced with 100

    df[col] = np.where(df[col] < 0, 0, df[col])
    # Values less than 0 replaced with 0

print("\nAfter Handling Missing Values & Inconsistencies:\n", df)
# Output: Displays cleaned dataset after fixing null and invalid values


# DETECT OUTLIERS (IQR)


def detect_outliers(column):
    # Function to calculate outlier boundaries using IQR method

    Q1 = column.quantile(0.25)
    # First Quartile (25%)

    Q3 = column.quantile(0.75)
    # Third Quartile (75%)

    IQR = Q3 - Q1
    # Interquartile Range

    lower = Q1 - 1.5 * IQR
    # Lower boundary for outliers

    upper = Q3 + 1.5 * IQR
    # Upper boundary for outliers

    return lower, upper
    # Returns lower and upper limits


for col in ["Maths", "Science", "English", "Attendance (%)"]:
    lower, upper = detect_outliers(df[col])
    # Detects outlier boundaries

    print(f"{col}: Lower={lower}, Upper={upper}")
    # Output: Displays outlier range for each column


# HANDLE OUTLIERS
# Using Winsorization
for col in ["Maths", "Science", "English", "Attendance (%)"]:

    lower, upper = detect_outliers(df[col])
    # Gets lower and upper limits

    df[col] = np.clip(df[col], lower, upper)
    # Replaces extreme values outside range with nearest boundary value

print("\nAfter Outlier Handling:\n", df)
# Output: Displays dataset after handling outliers


# DATA TRANSFORMATION
# Min-Max Normalization (0–1 scaling)

df["Attendance_Normalized"] = (df["Attendance (%)"] - df["Attendance (%)"].min()) / (
    df["Attendance (%)"].max() - df["Attendance (%)"].min()
)
# Converts Attendance values between 0 and 1

print("\nAfter Transformation:\n", df)
# Output: Shows normalized attendance column added


# VISUALIZATION

plt.figure()
sns.boxplot(data=df[["Maths", "Science", "English", "Attendance (%)"]])
# Output: Boxplot showing spread and outliers after cleaning

plt.title("Boxplot After Outlier Handling")
plt.show()


plt.figure()
sns.histplot(df["Attendance_Normalized"], kde=True)
# Output: Histogram showing normalized attendance distribution

plt.title("Normalized Attendance")
plt.show()


# FINAL OUTPUT
print("\nFinal Clean Dataset:\n", df)
# Output: Displays final cleaned and transformed dataset


# ============================================================
# COMPLETE OUTPUT EXPLANATION
# ============================================================

# 1. Original Dataset Output
# Displays raw dataset containing:
# Missing values (NaN)
# Invalid values like 120, 300, 110
# Used before preprocessing starts.

# 2. Missing Values Output
# Shows total null values in each column.
# Example:
# Maths may contain 1 missing value.
# Science may contain 1 missing value.

# 3. Mean Imputation Output
# Missing numerical values replaced using average value.
# Example:
# Missing Maths mark replaced with average Maths marks.

# 4. Inconsistency Handling Output
# Values greater than 100 replaced with 100.
# Values less than 0 replaced with 0.
# Example:
# 120 becomes 100
# 300 becomes 100
# 110 becomes 100

# 5. Outlier Detection Output
# IQR method calculates lower and upper boundaries.
# Example:
# Maths: Lower=40, Upper=120
# Values outside this range are treated as outliers.

# 6. Outlier Handling Output
# Winsorization limits extreme values to nearest valid boundary.
# Prevents outliers from affecting analysis heavily.

# 7. Transformation Output
# Attendance_Normalized column added.
# Values converted between 0 and 1 using Min-Max Scaling.
# Example:
# Minimum attendance becomes 0
# Maximum attendance becomes 1

# 8. Boxplot Output
# Displays:
# Median
# Spread of marks
# Outliers after cleaning
# Helps visualize data distribution.

# 9. Histogram Output
# Shows normalized attendance distribution.
# KDE curve displays probability density smoothly.

# 10. Final Clean Dataset Output
# Displays fully cleaned dataset after:
# Missing value handling
# Inconsistency correction
# Outlier treatment
# Normalization

# Overall Aim:
# This program demonstrates complete Data Cleaning,
# Data Transformation, Outlier Detection,
# Outlier Handling, and Visualization
# using Student Performance Dataset.
