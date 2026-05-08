import pandas as pd

# Load the dataset
df_cars = pd.read_csv('car data.csv')

# Initial inspection
print(df_cars.head())
print(df_cars.info())
print(df_cars.describe())
print(df_cars.isnull().sum())