import pandas as pd
from scipy import stats
import numpy as np

# Read the CSV file
df = pd.read_csv('Student_Grades.csv')

# Display the contents of the DataFrame
print("Result of df.head()")
print(df)


# ADDED BY GIO FOR Z_SCORES
# Calculate Z-scores for the Math column
math_clean = df['MathScore'].dropna()  # Drop NaNs for Z-score calculation
z_scores_math = np.abs(stats.zscore(math_clean))  # Calculate Z-scores

threshold = 3  # Common threshold for Z-scores
outliers_indices = math_clean.index[z_scores_math > threshold]  # Get indices of outliers
outliers_z_math = df.loc[outliers_indices] 

print("\nOutliers detected in Math column (using Z-score):")
print(outliers_z_math)

# Calculate Z-scores for the English column
english_clean = df['EnglishScore'].dropna()  # Drop NaNs for Z-score calculation
z_scores_english = np.abs(stats.zscore(english_clean))  # Calculate Z-scores

threshold = 3  # Common threshold for Z-scores
outliers_indices = english_clean.index[z_scores_english > threshold]  # Get indices of outliers
outliers_z_english = df.loc[outliers_indices] 

print("\nOutliers detected in English column (using Z-score):")
print(outliers_z_english)

# Calculate Z-scores for the Science column
science_clean = df['ScienceScore'].dropna()  # Drop NaNs for Z-score calculation
z_scores_science = np.abs(stats.zscore(science_clean))  # Calculate Z-scores

threshold = 3  # Common threshold for Z-scores
outliers_indices = science_clean.index[z_scores_science > threshold]  # Get indices of outliers
outliers_z_science = df.loc[outliers_indices] 

print("\nOutliers detected in Science column (using Z-score):")
print(outliers_z_science)

# Calculate Z-scores for the Attendance column
attendance_clean = df['Attendance'].dropna()  # Drop NaNs for Z-score calculation
z_scores = np.abs(stats.zscore(attendance_clean))

threshold = 3  # Common threshold for Z-scores
outliers_indices = attendance_clean.index[z_scores > threshold]  # Get indices of outliers
outliers_z = df.loc[outliers_indices] 

print("\nOutliers detected in Attendance column (using Z-score):")
print(outliers_z)