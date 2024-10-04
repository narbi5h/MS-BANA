import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Student_Grades.csv')
print(df)

#DETECT OUTLIERS BASED ON IQR
Q1 = df['EnglishScore'].quantile(0.25)
Q3 = df['EnglishScore'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds for english score
lower_bound_english = Q1 - 1.5 * IQR
upper_bound_english = Q3 + 1.5 * IQR

Q1 = df['MathScore'].quantile(0.25)
Q3 = df['MathScore'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds for Math score
lower_bound_math = Q1 - 1.5 * IQR
upper_bound_math = Q3 + 1.5 * IQR

# Detect outliers
outliers = df[(df['EnglishScore'] < lower_bound_english) | (df['EnglishScore'] > upper_bound_english) | (df['MathScore'] < lower_bound_math) | (df['MathScore'] > upper_bound_math)]

print(f"\nOutliers detected in English column (using IQR):\n{outliers}")

#df_no_outliers = df[(df['EnglishScore'] >= lower_bound_english) & (df['EnglishScore'] <= upper_bound_english)]
df_no_outliers = df[df['Name']!= 'Charlie']


df_fill_no_outliers=df_no_outliers.fillna(df_no_outliers.median(numeric_only=True))


#CAPPING THE AGE AND FILLING MISSING DATA FOR AGE AND ATTENDANCE
Q1 = df_fill_no_outliers['Age'].quantile(0.25)
Q3 = df_fill_no_outliers['Age'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = (Q1 - 1.5 * IQR).astype(int)
upper_bound = (Q3 + 1.5 * IQR).astype(int)

df_fill_no_outliers['Age_capped'] = df['Age'].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))
df_fill_no_outliers['Age_capped'].skew()
df_fill_no_outliers=df_fill_no_outliers.fillna(df_fill_no_outliers.median(numeric_only=True))
#median_age=(df['Age_capped'].median())

#re-arrange columns and print
df_fill_no_outliers = df_fill_no_outliers[['StudentID', 'Name', 'Age_capped', 'MathScore', 'EnglishScore', 'ScienceScore','Attendance']]
print(df_fill_no_outliers)