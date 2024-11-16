import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


os.chdir('C:/Users/gio12/Documents/GitHub/MS-BANA/Midterms')

# Load the dataset
df=pd.read_csv('ME2_Dataset.csv')

#drop Customer_ID
df.drop(columns=['Customer_ID'], inplace=True)

# # Check for missing values in each column
# missing_values = df.isnull().sum()
# print(missing_values)

# # Check kurtosis of all numerical columns
# kurtosis = df.select_dtypes(include=[np.number]).kurtosis()
# print(kurtosis)

# # Check skewness of all numerical columns
# skewness = df.select_dtypes(include=[np.number]).skew()
# print(skewness)

# #create a histogram for age score
# plt.figure(figsize=(6,4))
# sns.histplot(data=df, x='Age', color='blue')
# plt.show()

# # Create a histogram for income
# plt.figure(figsize=(6,4))
# sns.histplot(data=df, x='Income', color='green')
# plt.show()

# # Create a histogram for Spending_score
# plt.figure(figsize=(6,4))
# sns.histplot(data=df, x='Spending_Score', color='green')
# plt.show()

# # Create a histogram for Satisfaction_Score
# plt.figure(figsize=(6,4))
# sns.histplot(data=df, x='Satisfaction_Score', color='green')
# plt.show()