# Step 1: Load Dataset

import pandas as pd

# Load the dataset
df = pd.read_csv('Sales_Age_Income.csv')

# Show the dataset
print("Original DataFrame:")
print(df)

# Step 3: Handling Outliers

# 3.1. Detect and Remove Outliers Using IQR (Interquartile Range)
    # Outliers in the Age column can be detected and removed using the IQR method.

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]

print(f"\nOutliers detected in Age column (using IQR):\n{outliers}")

# Remove outliers
df_no_outliers = df[(df['Age'] >= lower_bound) & (df['Age'] <= upper_bound)]

print("\nDataFrame after removing outliers:")
print(df_no_outliers)

#3.2. Capping Outliers (Setting Upper/Lower Bound on Outliers)
    #Instead of removing outliers, you can cap them at the upper or lower bounds.

# Cap the outliers in the Age column
df['Age_capped'] = df['Age'].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))

print("\nDataFrame after capping outliers in Age column:")
print(df[['Name', 'Age', 'Age_capped']])

#3.3. Detect and Remove Outliers Using Z-score
    # You can also use the Z-score method to detect and handle outliers. 
    # A Z-score greater than 3 indicates a potential outlier.

from scipy import stats
import numpy as np

# Calculate Z-scores for the Age column
z_scores = np.abs(stats.zscore(df['Age'].dropna()))  # dropna to ignore NaNs
threshold = 3  # Common threshold for Z-scores
outliers_z = df.iloc[(z_scores > threshold).values]

print("\nOutliers detected in Age column (using Z-score):")
print(outliers_z)

# Remove outliers based on Z-score
df_no_outliers_z = df.iloc[(z_scores <= threshold).values]

print("\nDataFrame after removing outliers (based on Z-score):")
print(df_no_outliers_z)


#Recap of Steps:
    # Removing missing values: Demonstrates how to remove rows or columns with missing data.
    # Imputing missing values: Shows how to fill missing values with methods like mean, forward fill, or backward fill.
    # Handling outliers: Demonstrates detecting and handling outliers using both the IQR method and Z-score.