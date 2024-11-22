import pandas as pd
from faker import Faker

# Initialize Faker
fake = Faker()

# Sample DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David']}
df = pd.DataFrame(data)

# Replace column values with random last names
df['Name'] = df['Name'].apply(lambda x: fake.last_name())

print(df)