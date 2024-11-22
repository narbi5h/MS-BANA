
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk

# Read the dataset
df = pd.read_csv('life.csv')


# Filter the dataframe to include only rows where the country is Afghanistan, Albania, or Italy
filtered_df = df[df['Country'].isin(['Afghanistan'])]

filtered_df = filtered_df[['Year', 'Life expectancy ']]



# Create a time series plot for life expectancy over the years
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Life expectancy ', data=filtered_df, marker='o')
plt.title('Life Expectancy Over the Years')
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.grid(True)
plt.show()

# # Function to remove IQR outliers for each sailmonth
# def remove_outliers_iqr(df, column):
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# # Remove outliers for each month
# df = df.groupby('month').apply(lambda x: remove_outliers_iqr(x, 'Trams')).reset_index(drop=True)

# # Check for non-linear correlation between 'month' and 'Trams' using Spearman's rank correlation
# spearman_corr, spearman_p_value = stats.spearmanr(df['month'], df['Trams'])
# print(f"Spearman's rank correlation: {spearman_corr}, p-value: {spearman_p_value}")

# # Convert 'month' to numeric for plotting
# df['month'] = pd.to_numeric(df['month'], errors='coerce')

# # Create cat plot with mean
# plt.figure(figsize=(10, 6))
# sns.catplot(x='month', y='Trams', kind='box', data=df, showmeans=True, height=6, aspect=1.5)
# plt.title('Box Plot of Trams by Month')
# plt.xlabel('Month')
# plt.ylabel('Trams')
# plt.show()


# # Select only the 'departure_date' and 'cost_assumption' columns
# t_test = df[['departure_date', 'cost_assumption']]

# print(t_test.head())

# # Example T-test: Comparing Income based on Gender
# group_male = df[df['Gender'] == 'Male']['Spending_Score']
# group_female = df[df['Gender'] == 'Female']['Spending_Score']
# t_stat, p_value = stats.ttest_ind(group_male, group_female)
# print(f'T-Test Results: t-statistic = {t_stat}, p-value = {p_value}')
# print("Conclusion: ", "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis")

# Example F-test: Comparing variances of Spending Score between Marital Status groups
# # print(df['Education_Level'].unique())
# HS = df[df['Education_Level'] == 'High School']['Satisfaction_Score']
# BS = df[df['Education_Level'] == 'Bachelor']['Satisfaction_Score']
# MS = df[df['Education_Level'] == 'Master']['Satisfaction_Score']
# PHD = df[df['Education_Level'] == 'PhD']['Satisfaction_Score']
# f_stat, f_p_value = stats.levene(HS, BS, MS, PHD)
# print(f'F-Test Results: F-statistic = {f_stat}, p-value = {f_p_value}')
# print("Conclusion: ", "Reject null hypothesis" if f_p_value < 0.05 else "Fail to reject null hypothesis")

# # Example Chi-Squared Test: Testing relationship between Customer_Type and Product_Category
# contingency_table = pd.crosstab(df['Education_Level'], df['Marital_Status'])
# chi2_stat, chi2_p_val, dof, expected = stats.chi2_contingency(contingency_table)
# print(f'Chi-Squared Test Results: chi2_stat = {chi2_stat}, p-value = {chi2_p_val}')
# print("Conclusion: ", "Reject null hypothesis" if chi2_p_val < 0.05 else "Fail to reject null hypothesis")

# # Visualization: Boxplot for Income based on Gender
# sns.boxplot(x='Gender', y='Income', data=df)
# plt.title('Income Distribution by Gender')
# plt.show()
