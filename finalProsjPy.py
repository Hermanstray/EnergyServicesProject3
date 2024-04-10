
#%%
import pandas as pd # data science library o manipulate data
import numpy as np # mathematical library to manipulate arrays and matrices
import matplotlib.pyplot as plt # visualization library
import seaborn as sb #visualization library specific for data science, based on matplotlib 

# Read data and add Datetime
df_data = pd.read_csv('merged_data_2015-2019.csv') # loads a csv file into a dataframe

df_data['date'] = pd.to_datetime(df_data['date']) # create a new column 'data time' of datetime type
df_data = df_data.set_index('date',  drop=True) # make 'datetime' into index

#%%
# Adding some features

df_data['month'] = df_data.index.month

df_data.head()

#%%
# Splitting data into training and test data
from sklearn.model_selection import train_test_split

# Drop rows with NaN values
df_data.dropna(inplace=True)

Z=df_data.values
Y=Z[:,0]
X=Z[:,[1,2,3,4]]

#by default, it chooses randomly 75% of the data for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X,Y)
#X2_train, X2_test, y2_train, y2_test = train_test_split(X2,Y)

print(X_train)
print(y_train)

#%%
# Choosing Random Forest Regressor to create regression model

from sklearn import  metrics
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)

#%%
#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MBE_RF=np.mean(y_test-y_pred_RF) #here we calculate MBE
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
NMBE_RF=MBE_RF/np.mean(y_test)
print(MAE_RF,MBE_RF,MSE_RF,RMSE_RF,cvRMSE_RF,NMBE_RF)

#%%
# Save model

import pickle

#save LR model
with open('RF_model_final_prosj.pkl','wb') as file:
    pickle.dump(RF_model, file)

#%%
#Load LR model

with open('RF_model_final_prosj.pkl','rb') as file:
    RF_model=pickle.load(file)

y2_pred_RF = RF_model.predict(X_test)

plt.plot(y_test)
plt.plot(y2_pred_RF)
plt.show()
plt.scatter(y_test,y2_pred_RF)