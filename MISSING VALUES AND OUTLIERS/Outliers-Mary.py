import pandas as pd
from scipy import stats
import numpy as np

# A function that takes in a data frame and prints descriptive stats
def get_descriptive_stats(data_frame):
    for column in data_frame:
        data = data_frame[column]
        
        # Calculate descriptive statistics
        mean = data.mean()
        median = data.median()
        maximum = data.max()
        minimum = data.min()
        variance = data.var()
        std_dev = data.std()
        skewness = data.skew()
        kurtosis = data.kurtosis()

        # Display the results
        print("FOR", column.upper())
        print(f'Mean: {mean}')
        print(f'Median: {median}')
        print(f'Maximum: {maximum}')
        print(f'Minimum: {minimum}')
        print(f'Variance: {variance}')
        print(f'Standard Deviation: {std_dev}')
        print(f'Skewness: {skewness}')
        print(f'Kurtosis: {kurtosis}')
        print("")

# Read the CSV file
df = pd.read_csv('Student_Grades.csv')

# Display the contents of the DataFrame
print("Result of df.head()")
print(df.head())

"""
We have three dataframes below:
    ***num_cats*** is the original dataframe, but only with the columns we need to analyze,
    such as MathScore, EnglishScore, ScienceScore, and Attendance. It has unaltered data, 
    meaning it still has empty cells.

    ***forward_filled_df*** fills in empty cells by taking the last known value from a 
    previous row.

    ***backward_filled_df*** fills in empty cells by taking the next value from a subsequent 
    row. 
"""

# Original dataframe, but only with MathScore, EnglishScore, ScienceScore, and Attendance colums
num_cats = df.iloc[:, 3:]  # Select all rows and columns starting from index 3
print("\nOriginal DataFrame, with only columns we need:")
print(num_cats)
print("")

# 1) RUN DESCRIPTIVE STATS ON ALL VARIABLES
print("1) Run Descriptive Statistics on All Variables.:")
get_descriptive_stats(num_cats)

print('''2) Use different techniques (mean, median, mode, forward fill, backward fill) to 
      impute missing values for MathScore, EnglishScore, ScienceScore, and Attendance. 
      Discuss the impact of removing rows with missing values versus imputing the missing values.''')

# FORWARD FILL -------------------------------------------------------------------
forward_filled_df = num_cats.copy()  # Make a copy of the numerical categories
forward_filled_df = forward_filled_df.fillna(method='ffill')  # Apply forward fill
# Display the results
print("\nForward Filled DataFrame:")
print(forward_filled_df)

# BACKWARD FILL -----------------------------------------------------------------
backward_filled_df = num_cats.copy()  # Make a copy of the numerical categories
backward_filled_df = backward_filled_df.fillna(method='bfill')  # Apply backward fill
# Display the results
print("\nBackward Filled DataFrame:")
print(backward_filled_df)


# MEDIAN FILL -------------------------------------------------------------------
median_df = num_cats.copy()  # Make a copy of the numerical categories
median_df = median_df.fillna(median_df.median(numeric_only=True))  # Apply forward fill
# Display the results
print("\nForward Filled DataFrame:")
print(median_df)

# Calculate descriptive statistics for each dataframe
print("")
print("STATS FROM IMPUTING THE MISSING VALUES WITH FWD FILL:")
get_descriptive_stats(forward_filled_df)
print("")

print("")
print("STATS FROM IMPUTING THE MISSING VALUES WITH BWD FILL:")
get_descriptive_stats(backward_filled_df)
print("")

print("")
print("STATS FROM IMPUTING THE MISSING VALUES WITH MEDIAN FILL:")
get_descriptive_stats(median_df)
print("")



# 3) HANDLE OUTLIERS
print("""
3) Handle Outliers: Detect and remove/cap the outliers in the Age column.
Use IQR and Z-score methods to detect outliers in MathScore or other score columns.
    """)

print("Age column before change:")
print(df['Age'])
print("")

# Calculate the Interquartile Range (IQR) for Age column
quart_1 = df['Age'].quantile(0.25)  # 25th percentile (Q1)
quart_3 = df['Age'].quantile(0.75)  # 75th percentile (Q3)
iqr = quart_3 - quart_1  # Interquartile range

# Display IQR
print("Calculate IQR for 'AGE':")
print(f"Q1 (25th percentile): {quart_1}")
print(f"Q3 (75th percentile): {quart_3}")
print(f"IQR (Interquartile Range): {iqr}")

# Define bounds for outliers
lower_bound = quart_1 - 1.5 * iqr
upper_bound = quart_3 + 1.5 * iqr
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")

