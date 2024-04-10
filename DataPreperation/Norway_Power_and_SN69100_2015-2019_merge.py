#%%
import pandas as pd

power_data = pd.read_csv('Norway_Power_2015-2019_daily.csv')
weather_data = pd.read_csv('SN69100_2015-2019_daily.csv')

# Merge the two datasets on the 'date' column
merged_data = pd.merge(power_data, weather_data, on='date')

# Save the merged data to a new CSV file
merged_data.to_csv('merged_data_2015-2019.csv', index=False)
