#%%
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('Norway_Power_2015-2019_daily.csv')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['power_MW'])
plt.title('Daily Power Consumption')
plt.xlabel('Date')
plt.ylabel('Power (MW)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()