#!/usr/bin/env python
# coding: utf-8

# # #Step1:Importing Data set

# In[1]:


import pandas as pd
sw=pd.read_csv('/Users/vinaybhore/Project/MSFT.csv')
print("student data read successfully ")


# In[2]:


sw.head(n=3)


# In[3]:


sw.tail(n=3)


# In[4]:


get_ipython().system('pip install pandas_datareader')


# In[5]:


import pandas_datareader as pdr


# In[6]:


import datetime


# In[7]:


tsla = pdr.get_data_yahoo('TSLA', 
                          start=datetime.datetime(2004, 1, 1), 
                          end=datetime.datetime(2019, 9, 15))


# In[8]:


aapl = pdr.get_data_yahoo('AAPL', 
                          start=datetime.datetime(2004, 1, 1), 
                          end=datetime.datetime(2019, 9, 15))


# In[9]:


amzn = pdr.get_data_yahoo('AMZN', 
                          start=datetime.datetime(2004, 1, 1), 
                          end=datetime.datetime(2019, 9, 15))


# In[10]:


type(tsla),type(aapl)


# In[11]:


tsla.head(n=2)


# In[12]:


amzn.tail(n=3)


# In[13]:


amzn.describe()


# In[14]:


amzn.columns


# In[15]:


tsla.index,amzn.index


# In[16]:


amzn.shape


# # Time Series Data

# In[17]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# import matplotlib.dates as mdates
# 

# In[19]:


plt.plot(amzn.index, amzn['Adj Close'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.grid(True)
plt.xticks(rotation=90)
plt.show()


# In[33]:


#Subplots
f, ax = plt.subplots(2, 2, figsize=(10,10), sharex=True,sharey=True)
f.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
f.gca().xaxis.set_major_locator(mdates.MonthLocator())

tsla_18 = tsla.loc[pd.Timestamp('2019-12-01'):pd.Timestamp('2018-12-31')]
ax[0,0].plot(tsla.index, tsla['Adj Close'], color='r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('TSLA');


aapl_18 = aapl.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2018-12-31')]
ax[1,0].plot(aapl.index, aapl['Adj Close'], color='b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('AAPL');

amzn_18 = amzn.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2018-12-31')]
ax[1,1].plot(amzn.index, amzn['Adj Close'], color='y')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation=90)
ax[1,1].set_title('AMZN');



# In[ ]:


#Subplots
f, ax = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)
f.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
f.gca().xaxis.set_major_locator(mdates.MonthLocator())

tsla_18 = tsla.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2018-12-31')]
ax[0,0].plot(tsla_18.index, tsla_18['Adj Close'], '.', color='r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('TESLA');

amzn_18 = amzn.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2018-12-31')]
ax[0,1].plot(amzn_18.index, amzn_18['Adj Close'], '.' ,color='g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('AMAZON');

aapl_18 = aapl.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2018-12-31')]
ax[1,0].plot(aapl_18.index, aapl_18['Adj Close'], '.' ,color='b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('APPLE');


# # Resampling(quarterly)

# In[34]:


monthly_amzn_18 = amzn_18.resample('4M').mean()
plt.scatter(monthly_amzn_18.index, monthly_amzn_18['Adj Close'])
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()


# In[36]:


#Subplots
f, ax = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

monthly_tsla_18 = tsla_18.resample('4M').mean()
ax[0,0].scatter(monthly_tsla_18.index, monthly_tsla_18['Adj Close'], color='y')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('TESLA');

monthly_aapl_18 = aapl_18.resample('4M').mean()
ax[0,1].scatter(monthly_aapl_18.index, monthly_aapl_18['Adj Close'], color='g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('APPLE');

monthly_amzn_18 = amzn_18.resample('4M').mean()
ax[1,0].scatter(monthly_amzn_18.index, monthly_amzn_18['Adj Close'], color='b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('AMAZON');


# # Step 2:Resampling(Weekly)

# In[37]:


amzn_19 = amzn.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]


# In[38]:


weekly_amzn_19 = amzn_19.resample('W').mean()
weekly_amzn_19.head()


# In[ ]:


plt.plot(weekly_amzn_19.index, weekly_amzn_19['Adj Close'], '-o')
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()


# In[39]:


#Subplots
amzn_19 = amzn.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_amzn_19 = amzn_19.resample('W').mean()

aapl_19 = aapl.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_aapl_19 = aapl_19.resample('W').mean()

tsla_19 = tsla.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_tsla_19 = tsla_19.resample('W').mean()

f, ax = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)
ax[0,0].plot(weekly_amzn_19.index, weekly_amzn_19['Adj Close'], '-o', color='r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('AMAZON');

ax[0,1].plot(weekly_aapl_19.index, weekly_aapl_19['Adj Close'], '-o',color='g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('APPLE');

ax[1,0].plot(weekly_tsla_19.index, weekly_tsla_19['Adj Close'],'-o', color='b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('TESLA');


# In[40]:



amzn['diff'] = amzn['Open'] - amzn['Close']
amzn_diff = amzn.resample('W').mean()
amzn_diff.tail(10)


# In[41]:


plt.scatter(amzn_diff.loc['2019-01-01':'2019-09-15'].index, amzn_diff.loc['2019-01-01':'2019-09-15']['diff'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()


# In[43]:



#Subplots
tsla['diff'] = tsla['Open'] - tsla['Close']
tsla_diff = tsla.resample('W').mean()

aapl['diff'] = aapl['Open'] - aapl['Close']
aapl_diff = aapl.resample('W').mean()

amzn['diff'] = amzn['Open'] - amzn['Close']
amzn_diff = amzn.resample('W').mean()


f, ax = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

ax[0,0].scatter(tsla_diff.loc['2019-01-01':'2019-09-15'].index, tsla_diff.loc['2019-01-01':'2019-09-15']['diff']
, color='r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('TESLA');

ax[0,1].scatter(aapl_diff.loc['2019-01-01':'2019-09-15'].index, aapl_diff.loc['2019-01-01':'2019-09-15']['diff']
, color='g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('APPLE');

ax[1,0].scatter(amzn_diff.loc['2019-01-01':'2019-09-15'].index, amzn_diff.loc['2019-01-01':'2019-09-15']['diff']
, color='b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('AMAZON');


# # Step3:moving windows

# # Daily percentage

# In[44]:



daily_close_amzn = amzn[['Adj Close']]

# Daily returns
daily_pct_change_amzn = daily_close_amzn.pct_change()

# Replace NA values with 0
daily_pct_change_amzn.fillna(0, inplace=True)

daily_pct_change_amzn.head()


# In[45]:



daily_pct_change_amzn.hist(bins=50)

# Show the plot
plt.show()


# In[46]:


daily_close_tsla = tsla[['Adj Close']]

# Daily returns
daily_pct_change_tsla = daily_close_tsla.pct_change()

# Replace NA values with 0
daily_pct_change_tsla.fillna(0, inplace=True)

daily_close_tsla = tsla[['Adj Close']]

# Daily returns
daily_pct_change_tsla = daily_close_tsla.pct_change()

# Replace NA values with 0
daily_pct_change_tsla.fillna(0, inplace=True)

daily_close_amzn = amzn[['Adj Close']]

# Daily returns
daily_pct_change_amzn = daily_close_amzn.pct_change()

# Replace NA values with 0
daily_pct_change_amzn.fillna(0, inplace=True)

daily_close_amzn = amzn[['Adj Close']]

# Daily returns
daily_pct_change_amzn = daily_close_amzn.pct_change()

# Replace NA values with 0
daily_pct_change_amzn.fillna(0, inplace=True)

daily_pct_change_amzn.head()


# In[47]:



import seaborn as sns
sns.set()


# In[52]:


import seaborn as sns
# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(12, 7))

# Plot a simple histogram with binsize determined automatically
sns.distplot(daily_pct_change_tsla['Adj Close'], color="b", ax=axes[0, 0], axlabel='TESLA');

# Plot a kernel density estimate and rug plot
sns.distplot(daily_pct_change_amzn['Adj Close'], color="r", ax=axes[0, 1], axlabel='AMAZON');

# Plot a filled kernel density estimate
#sns.distplot(daily_pct_change_aapl['Adj Close'], color="b", ax=axes[1, 1], axlabel='APPLE');

# Plot a historgram and kernel density estimate
#sns.distplot(daily_pct_change_amzn['Adj Close'], color="m", ax=axes[1, 1], axlabel='AMAZON');


# # 4.volatality

# In[53]:


import numpy as np


# In[54]:



min_periods = 75 

# Calculate the volatility
vol = daily_pct_change_amzn.rolling(min_periods).std() * np.sqrt(min_periods) 

vol.fillna(0,inplace=True)

vol.tail()


# In[55]:


# Plot the volatility
vol.plot(figsize=(10, 8))

# Show the plot
plt.show()


# # #rolling means trends and seasonality

# In[57]:



amzn_adj_close_px = amzn['Adj Close']
# Short moving window rolling mean
amzn['42'] = amzn_adj_close_px.rolling(window=40).mean()

# Long moving window rolling mean
amzn['252'] = amzn_adj_close_px.rolling(window=252).mean()

# Plot the adjusted closing price, the short and long windows of rolling means
amzn[['Adj Close', '42', '252']].plot(title="AMAZON")

# Show plot
plt.show()

tsla_adj_close_px = tsla['Adj Close']
# Short moving window rolling mean
tsla['42'] = tsla_adj_close_px.rolling(window=40).mean()

# Long moving window rolling mean
tsla['252'] = tsla_adj_close_px.rolling(window=252).mean()

# Plot the adjusted closing price, the short and long windows of rolling means
tsla[['Adj Close', '42', '252']].plot(title="TESLA")

# Show plot
plt.show()


# In[61]:


amzn.loc['2019-01-01':'2019-06-15'][['Adj Close', '42', '252']].plot(title="AMAZON in 2019");
tsla.loc['2019-01-01':'2019-09-15'][['Adj Close', '42', '252']].plot(title="TESLA in 2019");


# In[ ]:




