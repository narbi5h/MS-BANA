import pandas as pd
from scipy import stats
import numpy as np

# Read the CSV file
df = pd.read_csv('Student_Grades.csv')

# Display the contents of the DataFrame
print("Result of df")
print(df)


#DETECT OUTLIERS BASED ON IQR FOR MATH SCORE
Q1 = df['MathScore'].quantile(0.25)
Q3 = df['MathScore'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds for Math score
lower_bound_math = Q1 - 1.5 * IQR
upper_bound_math = Q3 + 1.5 * IQR

# Detect outliers
outliers_math = df[(df['MathScore'] < lower_bound_math) | (df['MathScore'] > upper_bound_math)]

print(f"\nOutliers detected in Math column (using IQR):\n{outliers_math}")

#DETECT OUTLIERS BASED ON IQR 
Q1 = df['EnglishScore'].quantile(0.25)
Q3 = df['EnglishScore'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds for english score
lower_bound_english = Q1 - 1.5 * IQR
upper_bound_english = Q3 + 1.5 * IQR

outliers_english = df[(df['EnglishScore'] < lower_bound_english) | (df['EnglishScore'] > upper_bound_english)]

print(f"\nOutliers detected in English column (using IQR):\n{outliers_english}")


#DETECT OUTLIERS BASED ON IQR FOR SCIENCE SCORE
Q1 = df['ScienceScore'].quantile(0.25)
Q3 = df['ScienceScore'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds for science score
lower_bound_science = Q1 - 1.5 * IQR
upper_bound_science = Q3 + 1.5 * IQR

# Detect outliers
outliers_science = df[(df['ScienceScore'] < lower_bound_science) | (df['ScienceScore'] > upper_bound_science)]

print(f"\nOutliers detected in Science column (using IQR):\n{outliers_science}")