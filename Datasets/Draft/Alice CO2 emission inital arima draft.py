#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import math as mt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import plotly
import plotly.express as px
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


url = "C:/Users/alice/OneDrive - The City College of New York/Fall 2021/Senior Project 2/Project/resedential.csv"


# In[3]:


data = pd.read_csv(url)


# In[4]:


data.info


# In[5]:


data


# In[6]:


newdata = data[["Category","residential million metric tons of carbon dioxide"]].rename(columns={'Category': "Year","residential million metric tons of carbon dioxide": "CO2"})


# In[7]:


newdata.head()


# In[8]:


newdata.index = pd.to_datetime(newdata.Year, format = '%Y')


# In[9]:


newdata.head()


# In[10]:


fig=plt.figure(1)
plot1 = fig.add_subplot(111)
plot1.set_xlabel('Year')
plot1.set_ylabel("CO2 Emission")
plot1.set_title("Resedential Co2 emissions by year")
plot1.plot("Year", "CO2", data = newdata)


# In[11]:


def TestStationaryPlot(ts):
    rol_mean = ts.rolling(window = 12, center = False).mean()
    rol_std = ts.rolling(window = 12, center = False).std()
    
    plt.plot(ts, color = 'blue',label = 'Original Data')
    plt.plot(rol_mean, color = 'red', label = 'Rolling Mean')
    plt.plot(rol_std, color ='black', label = 'Rolling Std')
    
    plt.xlabel('Time in Years')
    plt.ylabel('Total Emissions')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation', fontsize = 25)
    plt.show(block= False)
    
    
    


# In[12]:


TestStationaryPlot(newdata.CO2)


# In[13]:


def TestStationaryAdfuller(ts, cutoff = 0.01):
    ts_test = adfuller(ts, autolag = 'AIC')
    ts_test_output = pd.Series(ts_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    
    for key,value in ts_test[4].items():
        ts_test_output['Critical Value (%s)'%key] = value
    print(ts_test_output)
    
    if ts_test[1] <= cutoff:
        print("Strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root, hence it is stationary")
    else:
        print("Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
        


# In[14]:


TestStationaryAdfuller(newdata.CO2)


# In[15]:


p = d = q = range(0, 2) # Define the p, d and q parameters to take any value between 0 and 2
pdq = list(itertools.product(p, d, q)) # Generate all different combinations of p, q and q triplets


# In[16]:


print(pdq)


# In[17]:


newdata


# In[18]:


df = newdata["CO2"]


# In[19]:


df


# In[20]:


for param in pdq:
    try:
        mod = ARIMA(df,order=param)
        results = mod.fit()
        print('ARMA{} - AIC:{}'.format(param, results.aic))
        print(results.summary())
        print("----------------------------------------------------------------------------------------------------")
    except:
        continue


# In[21]:


residuals = pd.DataFrame(results.resid)
residuals.plot()


# In[22]:


residuals.plot(kind='kde')


# In[23]:


print(residuals.describe())


# In[24]:


X = df.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = mt.sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


# In[ ]:




