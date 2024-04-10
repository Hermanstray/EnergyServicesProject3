import pandas as pd

# Load the CSV file, assuming it's named 'data.csv'
df = pd.read_csv('NorwayMeteoData.csv')

# Remove the first column
df = df.drop(df.columns[0], axis=1)

# Filter the dataframe to keep only rows with sourceId = 'SN69100'
df_filtered = df[df['sourceId'] == 'SN69100']

# Save the filtered data back to a new CSV file, if you want
df_filtered.to_csv('SN69100_daily.csv', index=False)