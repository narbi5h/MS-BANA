import numpy as np
import pandas as pd



# Load data
golf_df = pd.read_csv('golf.csv') # importing the csv file

print(golf_df.head())

# Remove 'Rank' and 'Player' columns
golf_df = golf_df.drop(columns=['Rank', 'Player'])


#Renaming Columns
golf_df.columns = [s.strip().replace(' ', '_') for s in golf_df.columns] # all columns
# Remove dollar signs and commas from 'Earnings' column and convert to numeric
golf_df['Earnings'] = golf_df['Earnings'].replace({'\$': '', ',': ''}, regex=True).astype(float)



# Print the skew of all the columns
df_filled_mean = golf_df.fillna(golf_df.mean(numeric_only=True))






# Descriptive statistics
print(df_filled_mean.describe()) # show summary statistics for each column


df = pd.DataFrame(df_filled_mean)
#print(df)

print(golf_df.describe())

print(df_filled_mean.describe())

#Finding Correlation
correlation_matrix = df.corr()
correlation_matrix.to_csv('golf_correlation_matrix.csv')

print(correlation_matrix)

# Visualize the finding
import seaborn as sn
import matplotlib.pyplot as plt

sn.heatmap(correlation_matrix, annot=True)
plt.show()

