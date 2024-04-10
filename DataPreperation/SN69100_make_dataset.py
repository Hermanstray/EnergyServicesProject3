#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming your CSV data is in a file named 'data.csv'
df = pd.read_csv('SN69100_daily.csv')

# Convert day, month, year to a datetime type and set as index
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df.set_index('date', inplace=True)

# Drop the original day, month, and year columns, along with 'sourceId', 'latitude', and 'longtitude'
df.drop(['day', 'month', 'year', 'sourceId', 'latitude', 'longtitude', 'max(air_temperature P1D)', 'max(relative_humidity P1D)', 'max(wind_speed P1D)', 'sum(precipitation_amount P1D)'], axis=1, inplace=True)

# Define a dictionary with old column names as keys and new column names as values
rename_dict = {
    'mean(air_temperature P1D)': 'air_temperature',
    'mean(relative_humidity P1D)': 'relative_humidity',
    'mean(wind_speed P1D)': 'wind_speed'
}
# Rename the columns
df.rename(columns=rename_dict, inplace=True)

# Remove the outliers
# Calculate the first quartile
Q1 = df.quantile(0.25)

# Calculate the third quartile
Q3 = df.quantile(0.75)

# Calculate the IQR
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify the outliers
outliers = (df < lower_bound) | (df > upper_bound)

# Replace the outliers with NaN values
df_outliers_removed = df.mask(outliers, np.nan)

# Interpolate the NaN values
df = df.interpolate(method='time') # Remove NaN with interpolate

# Filter the DataFrame to include data from 2015 to 2019
df1 = df.loc['2015-01-01':'2019-12-31']
df1.to_csv('SN69100_2015-2019_daily.csv', index=True)

print(df1.info())

# Filter the DataFrame to include data from 2020
df2 = df.loc['2020-01-01':'2020-12-31']
df2.to_csv('SN69100_2020_daily.csv', index=True)

print(df2.info())
