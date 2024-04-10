#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming your data is in 'your_data.csv'
df = pd.read_csv('Norway_Power_2015-01-01_to_2020-07-31_hourly.csv')

# Remove the 'end_time' column
df.drop('end_time', axis=1, inplace=True)

# Convert 'start_time' to datetime objects, taking into account the timezone
df['datetime'] = pd.to_datetime(df['start_time'], utc=True)

# Set 'start_time' as the index
df.set_index('datetime', inplace=True) # Hourly index

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
df = df.mask(outliers, np.nan)

# Interpolate the NaN values
df = df.interpolate(method='time') # Remove NaN with interpolate

# Resample the data to daily frequency, summing up the 'power_MW'
df = df.resample('D').mean().round(3)

# Change index from datetime to date
df['date'] = pd.to_datetime(df.index.date)
df.set_index('date', inplace=True)

# print(df)

# Filter the DataFrame to include data from 2015 to 2019
df1 = df.loc['2015-01-01':'2019-12-31']
df1.to_csv('Norway_Power_2015-2019_daily.csv', index=True)

print(df1.info())

# Filter the DataFrame to include data from 2020
df2 = df.loc['2020-01-01':'2020-12-31']
df2.to_csv('Norway_Power_2020_daily.csv', index=True)

print(df2.info())