#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming your CSV data is in a file named 'data.csv'
df = pd.read_csv('SN69100_2015-2019_daily.csv')

# Convert day, month, year to a datetime type and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

print(df.describe()) # Length is 1826, which is the number of days between 2015-01-01 and 2019-12-31, considering 2016 is a leap year

# Plotting all the data
plt.figure(figsize=(10, 6))  # Set the figure size for better readability
df.plot()
plt.title('Weather Data for SN69100')
plt.xlabel('Date')
plt.ylabel('Measured Values')
plt.legend(loc='best')
plt.show()