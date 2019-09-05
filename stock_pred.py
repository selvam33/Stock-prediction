
# coding: utf-8

# In[33]:


# Import necessary packages
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


# In[3]:


df = pd.read_csv(r'C:\Users\ssamymuthu\Downloads\AAPL.csv')


# In[4]:


df.set_index('Date', inplace=True)
df.tail()


# In[5]:


df['Adj Close'].plot(label='AAPL', figsize=(16,8), title='Adjusted Closing Price', grid=True)


# In[7]:


window_size = 45 # Allow us to look at 45 days into the past
# Prepare the data so that we have 45 day windows and predict what the next day should be

# Get indices of access for the data
num_samples = len(df) - window_size
indices = np.arange(num_samples).astype(np.int)[:,None] + np.arange(window_size + 1).astype(np.int)


# In[8]:


data = df['Adj Close'].values[indices] # Create the 2D matrix of training samples


# In[9]:


X = data[:,:-1] # Each row represents 45 days in the past
y = data[:,-1] # Each output value represents the 46th day


# In[12]:


# Train and test split
split_fraction = 0.8
ind_split = int(split_fraction * num_samples)
X_train = X[:ind_split]
y_train = y[:ind_split]
X_test = X[ind_split:]
y_test = y[ind_split:]


# # Linear regression
# 

# In[13]:


#model
linear_reg = LinearRegression(n_jobs=-1)

#train
linear_reg.fit(X_train, y_train)

#predict
y_linear_pred_train = linear_reg.predict(X_train)
y_linear_pred = linear_reg.predict(X_test)


# # Linear Training Data

# In[14]:


# Plot what it looks like for the training data
df_linear = df.copy()
df_linear.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_linear = df_linear.iloc[window_size:ind_split] # Past 45 days we don't know yet
df_linear['Adj Close Train'] = y_linear_pred_train[:-window_size]
df_linear.plot(label='AAPL', figsize=(16,8), title='Adjusted Closing Price', grid=True)


# # Linear test data 

# In[15]:


# Same for the test
df_linear = df.copy()
df_linear.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_linear = df_linear.iloc[ind_split+window_size:] # Past 45 days we don't know yet
df_linear['Adj Close Test'] = y_linear_pred
df_linear.plot(label='AAPL', figsize=(16,8), title='Adjusted Closing Price', grid=True)


# # Lasso Regeression

# In[16]:


#Load model
lasso = Lasso(alpha=0.1)
#train
lasso.fit(X_train, y_train)

#Predict
y_pred_train_lasso = lasso.predict(X_train)
y_pred_lasso = lasso.predict(X_test)


# # Lasso training data

# In[17]:


# Plot what it looks like for the training data
df_lasso = df.copy()
df_lasso.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_lasso = df_lasso.iloc[window_size:ind_split] # Past 45 days we don't know yet
df_lasso['Adj Close Train'] = y_pred_train_lasso[:-window_size]
df_lasso.plot(label='AAPL', figsize=(16,8), title='Adjusted Closing Price', grid=True)


# # Lasso test data

# In[18]:


# Plot what it looks like for the training data
df_lasso = df.copy()
df_lasso.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_lasso = df_lasso.iloc[ind_split+window_size:] # Past 45 days we don't know yet
df_lasso['Adj Close Train'] = y_pred_lasso
df_lasso.plot(label='AAPL', figsize=(16,8), title='Adjusted Closing Price', grid=True)


# # Polynomial regression - Linear model Ridge

# In[82]:


#Load the Model
model = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', Ridge(fit_intercept=False))])

# Train
model.fit(X_train, y_train)

#predict
y_pred_train_model = model.predict(X_train)
y_pred_model = model.predict(X_test)


# In[94]:


poly_trained_data = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred_train_model})
poly_trained_data.head(10)


# # Polynomial - Train data

# In[83]:


# Plot what it looks like for the training data
df_model = df.copy()
df_model.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_model = df_model.iloc[window_size:ind_split] # Past 45 days we don't know yet
df_model['Adj Close Train'] = y_pred_train_model[:-window_size]
df_model.plot(label='AAPL', figsize=(16,8), title='Adjusted Closing Price', grid=True)


# In[96]:


poly_test_data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_model})
poly_test_data.head(10)


# # Polynomial - Test data

# In[84]:


# Plot what it looks like for the training data
df_model = df.copy()
df_model.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_model = df_model.iloc[ind_split+window_size:] # Past 45 days we don't know yet
df_model['Adj Close Train'] = y_pred_model
df_model.plot(label='AAPL', figsize=(16,8), title='Adjusted Closing Price', grid=True)


# # Train Data of the Three model

# In[67]:


df_train = df.copy()
df_train.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_train = df_train.iloc[window_size:ind_split] # Past 45 days we don't know yet
# Add in all of our methods
df_train['Adj Close Train Lasso'] = y_pred_train_lasso[:-window_size]
df_train['Adj Close Train Linear'] = y_linear_pred_train[:-window_size]
df_train['Adj Close Train Polynomial'] = y_pred_train_model[:-window_size]
# Plot the data now
df_train.plot(label='AMAT', figsize=(16,8), title='Adjusted Closing Price', grid=True)


# In[68]:


df_test = df.copy()
df_test.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_test = df_test.iloc[ind_split+window_size:] # Past 45 days we don't know yet
# Add in all of our methods
df_test['Adj Close Test Lasso'] = y_pred_lasso
df_test['Adj Close Test Linear'] = y_linear_pred
df_test['Adj Close Test Polynomial'] = y_pred_model
# Plot the data now
df_test.plot(label='AMAT', figsize=(16,8), title='Adjusted Closing Price', grid=True)


# In[69]:


num_days = 45 # Let's try and forecase the next 45 days or two years

# Get the last window_size (45) days
data_seed = df['Adj Close'].values[-window_size:][None]

input_values = {"lasso": data_seed, "linear": data_seed, "poly": data_seed}
values = {"lasso": [], "linear": [], "poly": []}
for i in range(num_days): # For each day...
    # Predict the next price given the previous N prices
    values["lasso"].append(lasso.predict(input_values["lasso"])[0])
    values["linear"].append(linear_reg.predict(input_values["linear"])[0])
    values["poly"].append(linear_reg.predict(input_values["poly"])[0])
    # Dump the oldest price and put the newest price at the end
    for v in input_values:
        val = input_values[v]
        val = np.insert(val, -1, values[v][-1], axis=1)
        val = np.delete(val, 0, axis=1)
        input_values[v] = val.copy()

# Convert all to NumPy arrays
for v in input_values:
    values[v] = np.array(values[v])
    


# In[70]:


from datetime import timedelta, datetime
last_date = datetime.strptime(df.index[-1], '%Y-%m-%d')
df_forecast = pd.DataFrame()
df_forecast["Lasso"] = values["lasso"]
df_forecast["Linear"] = values["linear"]
df_forecast["Polynomial Ridge"] = values["poly"]
df_forecast.index = pd.date_range(start=last_date, periods=num_days)
df_forecast.plot(label='AAPL', figsize=(16,8), title='Forecasted Adjusted Closing Price', grid=True)


# In[106]:


df.describe()


# # Poly error calculation

# In[100]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_model))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_model))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_model)))


# #  Linear error calculation

# In[101]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_linear_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_linear_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_linear_pred)))


# # Lasso Error calculation

# In[102]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_lasso))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_lasso))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_lasso)))

