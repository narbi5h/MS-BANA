import os

# Set the workspace directory path
workspace_dir = ('\\Users\\gio12\\Documents\\GitHub\\MS-BANA')  # Replace with your actual workspace directory path

print(workspace_dir)

# Change the current working directory to the workspace directory
os.chdir(workspace_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Read the CSV file into a DataFrame
df = pd.read_csv('ABE_MODEL.csv')

# Drop rows where the SCORE column is null
df = df.dropna(subset=['cost_assumption'])
                       

# Apply label encoding to categorical features
label_encoders = {}
categorical_features = ['cruise', 'home', 'connectinggateway', 'arrivalcity']
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target variable
X = df[['cruise','home','connectinggateway','arrivalcity']]

y = df['cost_assumption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



model = DecisionTreeRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy_score(y_test, predictions)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
RMSE=np.sqrt(mse)

print(f"Mean Squared Error: {mae}")
print(f"R^2 Score: {r2}")
print(f"RMSE: {RMSE}")


residuals = y_test - predictions
plt.scatter(predictions, residuals)
plt.hlines(y=0, xmin=min(predictions), xmax=max(predictions), colors='r')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

