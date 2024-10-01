import pandas as pd
from scipy import stats
import numpy as np

# Read the CSV file
df = pd.read_csv('Student_Grades.csv')

# Display the contents of the DataFrame
print("Result of df.head()")
print(df.describe())