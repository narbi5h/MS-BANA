# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a DataFrame
healthData = pd.read_csv('healthData.csv')

# Step 3: Perform Descriptive Statistics
# -------------------------------------
# Descriptive statistics for numerical variables ('age' and 'salary')
# Including mean, median, mode, standard deviation, variance, min, max, range, percentiles, IQR, skewness, and kurtosis

# Descriptive statistics for 'age'
print("Descriptive statistics for 'age':")
print(healthData['age'].describe())
print("Median of 'age':", healthData['age'].median())
print("Range of 'age':", healthData['age'].max() - healthData['age'].min())
print("variance of 'age':", healthData['age'].var())
print("Mode of 'age':", healthData['age'].mode()[0])
print("Skewness of 'age':", healthData['age'].skew())
print("Kurtosis of 'age':", healthData['age'].kurtosis())
print("Percentiles for 'age':", healthData['age'].quantile([0.01, 0.10, 0.25, 0.75, 0.90, 0.99]))
IQR_age = healthData['age'].quantile(0.75) - healthData['age'].quantile(0.25)
print("IQR (Interquartile Range) of 'age':", IQR_age)

# Descriptive statistics for 'salary'
print("\nDescriptive statistics for 'salary':")
print(healthData['salary'].describe())
print("Median of 'salary':", healthData['salary'].median())
print("Range of 'salary':", healthData['salary'].max() - healthData['salary'].min())
print("variance of 'salary':", healthData['salary'].var())
print("Mode of 'salary':", healthData['salary'].mode()[0])
print("Skewness of 'salary':", healthData['salary'].skew())
print("Kurtosis of 'salary':", healthData['salary'].kurtosis())
print("Percentiles for 'salary':", healthData['salary'].quantile([0.01, 0.10, 0.25, 0.75, 0.90, 0.99]))
IQR_salary = healthData['salary'].quantile(0.75) - healthData['salary'].quantile(0.25)
print("IQR (Interquartile Range) of 'salary':", IQR_salary)

# Descriptive statistics for categorical variables
# Including mode, frequency distribution, and relative frequency

# Descriptive statistics for 'gender'
print("\nFrequency distribution for 'gender':")
print(healthData['gender'].value_counts(normalize=True))
print("Mode of 'Gender':")
print(healthData['gender'].mode())
print("Relative frequency distribution for 'smoking_habit':")
print(healthData['smoking_habit'].value_counts(normalize=True))
print("Mode of 'smoking_habit':")
print(healthData['smoking_habit'].mode())
print("Relative frequency distribution for 'work_out_habit':")
print(healthData['work_out_habit'].value_counts(normalize=True))
print("Mode of 'work_out_habit':")
print(healthData['work_out_habit'].mode())
print("Relative frequency distribution for 'heart_attack':")
print(healthData['heart_attack'].value_counts(normalize=True))
print("Mode of 'heart_attack':")
print(healthData['heart_attack'].mode())
print("Relative frequency distribution for 'education':")
print(healthData['education'].value_counts(normalize=True))
print("Mode of 'education':")
print(healthData['education'].mode())

#REPLACE MISSING WORK_OUT_HABIT VALUES TO NONE
healthData['work_out_habit'].fillna('None', inplace=True)

# Step 3: Visualizations
# ----------------------
# Bar chart for 'gender'
# healthData['gender'].value_counts().plot(kind='bar')
# plt.title('Gender Distribution')
# plt.xticks(rotation=0)
# plt.xlabel('Gender')
# plt.ylabel('Count')
# plt.show()

# Histogram for 'age'
# plt.hist(healthData['age'].dropna(), bins=20)
# plt.title('Age Distribution')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()

# Histogram for 'salary'
# plt.hist(healthData['salary'].dropna(), bins=20)
# plt.title('Salary Distribution')
# plt.xlabel('Salary')
# plt.ylabel('Frequency')
# plt.show()


# # Histogram for 'smoking_habit'
# healthData['smoking_habit'].value_counts().plot(kind='bar')
# plt.xticks(rotation=0)
# plt.title('Smoking Habit Distribution')
# plt.xlabel('Smoking Habit')
# plt.ylabel('Count')
# plt.show()

# # Histogram for 'work_out_habit'
# healthData['work_out_habit'].value_counts().plot(kind='bar')
# plt.xticks(rotation=0)
# plt.title('Work Out Habit Distribution')
# plt.xlabel('Work Out Habit')
# plt.ylabel('Count')
# plt.show()

# #Histogram for 'heart_attack'
# healthData['heart_attack'].value_counts().plot(kind='bar')
# plt.xticks(rotation=0)
# plt.title('Heart Attack Distribution')
# plt.xlabel('Heart Attack')
# plt.ylabel('Count')
# plt.show()

# # Histogram for 'education'
# healthData['education'].value_counts().plot(kind='bar')
# plt.xticks(rotation=0)
# plt.title('Education Distribution')
# plt.xlabel('Education')
# plt.ylabel('Count')
# plt.show()

# # Boxplot for 'age'
# plt.boxplot(healthData['age'].dropna())
# plt.title('Boxplot of Age')
# plt.ylabel('Age')
# plt.show()

# # # Boxplot for 'salary'
# plt.boxplot(healthData['salary'].dropna())
# plt.title('Boxplot of Salary')
# plt.ylabel('Salary')
# plt.show()

# # Step 4: Handle Missing Values and Outliers
# # ------------------------------------------
# # Impute missing values in 'age' and 'salary' using the median of each column
healthData_cleaned = healthData.copy()
healthData_cleaned['age'].fillna(healthData['age'].median(), inplace=True)
healthData_cleaned['salary'].fillna(healthData['salary'].median(), inplace=True)


# Impute missing values in categorical columns using the mode of each column

healthData_cleaned['gender'].fillna(healthData['gender'].mode(), inplace=True)
healthData_cleaned['smoking_habit'].fillna(healthData['smoking_habit'].mode(), inplace=True)
healthData_cleaned['work_out_habit'].fillna(healthData['work_out_habit'].mode(), inplace=True)
healthData_cleaned['heart_attack'].fillna(healthData['heart_attack'].mode(), inplace=True)
healthData_cleaned['education'].fillna(healthData['education'].mode(), inplace=True)


# Handle outliers in 'age' using the IQR method
Q1_age = healthData_cleaned['age'].quantile(0.25)
Q3_age = healthData_cleaned['age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
lower_bound_age = Q1_age - 1.5 * IQR_age
upper_bound_age = Q3_age + 1.5 * IQR_age
# Remove outliers from 'age'
healthData_cleaned['age'] = healthData_cleaned['age'].apply(lambda x: upper_bound_age if x > upper_bound_age else (lower_bound_age if x < lower_bound_age else x))
#healthData_cleaned = healthData_cleaned[(healthData_cleaned['age'] >= lower_bound_age) & (healthData_cleaned['age'] <= upper_bound_age)]

# # Handle outliers in 'salary' using the IQR method
# Q1_salary = healthData['salary'].quantile(0.25)
# Q3_salary = healthData['salary'].quantile(0.75)
# IQR_salary = Q3_salary - Q1_salary
# lower_bound_salary = Q1_salary - 1.5 * IQR_salary
# upper_bound_salary = Q3_salary + 1.5 * IQR_salary
# # Remove outliers from 'salary'
# healthData = healthData[(healthData['salary'] >= lower_bound_salary) & (healthData['salary'] <= upper_bound_salary)]

# Step 3 (Re-run): Perform Descriptive Statistics After Cleaning
# -------------------------------------------------------------
# Descriptive statistics for 'age' after cleaning
print("Descriptive statistics for 'age after cleaning':")
print(healthData_cleaned['age'].describe())
print("Median of 'age':", healthData_cleaned['age'].median())
print("Range of 'age':", healthData_cleaned['age'].max() - healthData['age'].min())
print("variance of 'age':", healthData_cleaned['age'].var())
print("Mode of 'age':", healthData_cleaned['age'].mode()[0])
print("Skewness of 'age':", healthData_cleaned['age'].skew())
print("Kurtosis of 'age':", healthData_cleaned['age'].kurtosis())
print("Percentiles for 'age':", healthData_cleaned['age'].quantile([0.01, 0.10, 0.25, 0.75, 0.90, 0.99]))
IQR_age = healthData_cleaned['age'].quantile(0.75) - healthData_cleaned['age'].quantile(0.25)
print("IQR (Interquartile Range) of 'age':", IQR_age)

# # Descriptive statistics for 'salary' after cleaning
print("\nDescriptive statistics for 'salary' after cleaning:")
print(healthData_cleaned['salary'].describe())
print("Median of 'salary':", healthData_cleaned['salary'].median())
print("Range of 'salary':", healthData_cleaned['salary'].max() - healthData_cleaned['salary'].min())
print("variance of 'salary':", healthData_cleaned['salary'].var())
print("Mode of 'salary':", healthData_cleaned['salary'].mode()[0])
print("Skewness of 'salary':", healthData_cleaned['salary'].skew())
print("Kurtosis of 'salary':", healthData_cleaned['salary'].kurtosis())
print("Percentiles for 'salary':", healthData_cleaned['salary'].quantile([0.01, 0.10, 0.25, 0.75, 0.90, 0.99]))
IQR_salary = healthData_cleaned['salary'].quantile(0.75) - healthData_cleaned['salary'].quantile(0.25)
print("IQR (Interquartile Range) of 'salary':", IQR_salary)

# Re-run Visualizations After Cleaning



# Histogram for 'age' after cleaning
# plt.hist(healthData_cleaned['age'].dropna(), bins=20)
# plt.title('Age Distribution after capping the Age outliers')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()


# Boxplot for 'age' after cleaning
# plt.boxplot(healthData_cleaned['age'].dropna())
# plt.title('Boxplot of Age After Cleaning')
# plt.ylabel('Age')
# plt.show()


# # Histogram for 'salary' after cleaning
# plt.hist(healthData_cleaned['salary'].dropna(), bins=20)
# plt.title('Salary Distribution After Cleaning')
# plt.xlabel('Salary')
# plt.ylabel('Frequency')
# plt.show()


# # Boxplot for 'salary' after cleaning
# plt.boxplot(healthData_cleaned['salary'].dropna())
# plt.title('Boxplot of Salary After Cleaning')
# plt.ylabel('Salary')
# plt.show()


##Bar chart for 'gender' after cleaning
# healthData_cleaned['gender'].value_counts().plot(kind='bar')
# plt.title('Gender Distribution After Cleaning')
# plt.xticks(rotation=0)
# plt.xlabel('Gender')
# plt.ylabel('Count')
# plt.show()

# Histogram for 'smoking_habit' after cleaning
# healthData_cleaned['smoking_habit'].value_counts().plot(kind='bar')
# plt.xticks(rotation=0)
# plt.title('Smoking Habit Distribution after imputation of missing values')
# plt.xlabel('Smoking Habit')
# plt.ylabel('Count')
# plt.show()


# Histogram for 'work_out_habit' after cleaning
# healthData_cleaned['work_out_habit'].value_counts().plot(kind='bar')
# plt.xticks(rotation=0)
# plt.title('Work Out Habit Distribution after imputation of missing values')
# plt.xlabel('Work Out Habit')
# plt.ylabel('Count')
# plt.show()


# Histogram for 'heart_attack' after cleaning
# healthData_cleaned['heart_attack'].value_counts().plot(kind='bar')
# plt.xticks(rotation=0)
# plt.title('Heart Attack Distribution after imputation of missing values')
# plt.xlabel('Heart Attack')
# plt.ylabel('Count')
# plt.show()

# # Histogram for 'education' after cleaning
# healthData_cleaned['education'].value_counts().plot(kind='bar')
# plt.xticks(rotation=0)
# plt.title('Education Distribution after imputation of missing values')
# plt.xlabel('Education')
# plt.ylabel('Count')
# plt.show()
