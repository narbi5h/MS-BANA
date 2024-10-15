# Step 1: Load Dataset

import pandas as pd

# Load the dataset
df = pd.read_csv('Sales_Age_Income.csv')

# Show the dataset
print("Original DataFrame:")
print(df)

#Step 2: Dealing with Missing Values

# 2.1. Remove Rows with Missing Values
    # Remove rows where any column has a missing value

df_dropped_rows = df.dropna()

print("\nDataFrame after removing rows with missing values:")
print(df_dropped_rows)

# 2.2. Remove Columns with Missing Values
    # Remove columns that contain any missing values

df_dropped_columns = df.dropna(axis=1)

print("\nDataFrame after removing columns with missing values:")
print(df_dropped_columns)

# 2.3. Impute Missing Values Using Mean (for Numerical Data)
    # Fill missing values in 'Age', 'Income', and 'Sales' with the mean of the respective columns

df_filled_mean = df.fillna(df.mean(numeric_only=True))

print("\nDataFrame after filling missing values with mean:")
print(df_filled_mean)

# 2.4. Impute Missing Values Using Forward Fill
    # Fill missing values using forward fill (fills with the previous row's value)

df_ffill = df.fillna(method='ffill')

print("\nDataFrame after forward filling missing values:")
print(df_ffill)

# 2.5. Impute Missing Values Using Backward Fill
    # Fill missing values using backward fill (fills with the next row's value)

df_bfill = df.fillna(method='bfill')

print("\nDataFrame after backward filling missing values:")
print(df_bfill)