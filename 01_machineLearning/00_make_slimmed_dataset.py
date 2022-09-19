#!/usr/bin/env python
# coding: utf-8

# # Dataset Creation
#
# This code relies on data from a [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv) that provides real-estate data. We will extract the data for sale price vs floor area.
#

# In[1]:


# Use Pandas for easy handling of CSV files.
import pandas as pd


# In[2]:


# Download data from:
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv
df = pd.read_csv('realestate_data.csv')
# Print the columns.
print(df.columns)


# In[3]:


# We want to select houses in reasonable condition.
df2 = df[df['OverallCond'] > 5]  # Clean up data a bit.


# In[4]:


# Only keep the 'SalePrice' and 'GrLivArea' (above ground living area) columns.
slim_df = df2[['SalePrice', 'GrLivArea']]


# In[5]:


# Save this data as a new CSV file.
slim_df.to_csv('slimmed_realestate_data.csv')


# In[6]:


# See a quick plot to verify this can be fit with linear regression.
slim_df.plot(x='SalePrice', y='GrLivArea', style='.')
