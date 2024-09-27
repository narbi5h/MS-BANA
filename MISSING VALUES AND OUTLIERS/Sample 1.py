import pandas as pd
import numpy as np

# Sample dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, np.nan, 35, 45, np.nan],
    'Salary': [50000, 60000, np.nan, 80000, 90000]
}

df = pd.DataFrame(data)

# Show the dataset with missing values
print("Original DataFrame:")
print(df)

# 1. Removing rows with missing values
df_dropped = df.dropna()
print("\nDataFrame after dropping missing values:")
print(df_dropped)

# 2. Filling missing values with a specific value (e.g., mean, median, or 0)
df_filled_mean = df.fillna(df.mean(numeric_only=True))
print("\nDataFrame after filling missing values with mean:")
print(df_filled_mean)

# 3. Imputation with forward fill or backward fill
df_ffill = df.fillna(method='ffill')
print("\nDataFrame after forward filling missing values:")
print(df_ffill)
