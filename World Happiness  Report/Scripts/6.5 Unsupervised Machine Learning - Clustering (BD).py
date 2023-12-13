#!/usr/bin/env python
# coding: utf-8

# # 6.5 Unspervised Machine Learning

# ## This script contains the following:
# 1. Importing libraries and data and renaming columns
# 2. The elbow technique
# 3. k-means clustering

# ## Importing libraries and data and renaming columns

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.cluster import KMeans # Here is where you import the k-means algorithm from scikit-learn.
import pylab as pl # PyLab is a convenience module that bulk imports matplotlib.


# In[2]:


# This option ensures the graphs you create are displayed in your notebook without the need to "call" them specifically.

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


path = '/Users/brad/Desktop/ACHIEVEMENT 6'


# In[4]:


# Import the  data

df = pd.read_csv(os.path.join(path, '/Users/brad/Desktop/ACHIEVEMENT 6/Original Data/archive (3)/2019.csv'))


# In[5]:


# Rename columns whose names are too long

df.rename(columns = {'Overall rank' : 'Rank', 'Healthy life expectancy': 'Life expectancy', 
                     'Freedom to make life choices': 'freedom of choice'},
                      inplace = True)


# In[21]:


df.columns


# ## 2. The Elbow Technique

# In[28]:


# Removing the categorical variable
df.drop('Country or region', axis=1, inplace=True)


# In[29]:


num_cl = range(1, 5) # Defines the range of potential clusters in the data.
kmeans = [KMeans(n_clusters=i) for i in num_cl] # Defines k-means clusters in the range assigned above.


# In[30]:


score = [kmeans[i].fit(df).score(df) for i in range(len(kmeans))] # Creates a score that represents 
# a rate of variation for the given cluster option.

score


# In[31]:


# Plot the elbow curve using PyLab.

pl.plot(num_cl,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()


# There is a large jump from two to three on the x-axis, but after that, the curve straightens out. This means that the optimal count for the clusters is three

# ## K-means clustering

# In[32]:


# Create the k-means object.

kmeans = KMeans(n_clusters = 3) 


# In[33]:


# Fit the k-means object to the data.

kmeans.fit(df)


# In[34]:


df['clusters'] = kmeans.fit_predict(df)


# In[35]:


df.head()


# In[36]:


df['clusters'].value_counts()


# In[37]:


# Plot the clusters for the "Score" and "GDP per capita" variables.

plt.figure(figsize=(12,8))
ax = sns.scatterplot(x=df['Score'], y=df['GDP per capita'], hue=kmeans.labels_, s=100) 
# Here, you're subsetting `X` for the x and y arguments to avoid using their labels. 
# `hue` takes the value of the attribute `kmeans.labels_`, which is the result of running the k-means algorithm.
# `s` represents the size of the points you want to see in the plot.

ax.grid(False) # This removes the grid from the background.
plt.xlabel('GDP per capita') # Label x-axis.
plt.ylabel('Score') # Label y-axis.
plt.show()


# In[38]:


# Plot the clusters for the "Score" and "Life expectancy" variables.

plt.figure(figsize=(12,8))
ax = sns.scatterplot(x=df['Score'], y=df['Life expectancy'], hue=kmeans.labels_, s=100) 
# Here, you're subsetting `X` for the x and y arguments to avoid using their labels. 
# `hue` takes the value of the attribute `kmeans.labels_`, which is the result of running the k-means algorithm.
# `s` represents the size of the points you want to see in the plot.

ax.grid(False) # This removes the grid from the background.
plt.xlabel('Life Expectancy') # Label x-axis.
plt.ylabel('Score') # Label y-axis.
plt.show()


# In[39]:


# Plot the clusters for the "Score" and "Social support" variables.

plt.figure(figsize=(12,8))
ax = sns.scatterplot(x=df['Score'], y=df['Life expectancy'], hue=kmeans.labels_, s=100) 
# Here, you're subsetting `X` for the x and y arguments to avoid using their labels. 
# `hue` takes the value of the attribute `kmeans.labels_`, which is the result of running the k-means algorithm.
# `s` represents the size of the points you want to see in the plot.

ax.grid(False) # This removes the grid from the background.
plt.xlabel('Social support') # Label x-axis.
plt.ylabel('Score') # Label y-axis.
plt.show()


# In[40]:


# Plot the clusters for the "Score" and "Freedom of choice" variables.

plt.figure(figsize=(12,8))
ax = sns.scatterplot(x=df['Score'], y=df['Life expectancy'], hue=kmeans.labels_, s=100) 
# Here, you're subsetting `X` for the x and y arguments to avoid using their labels. 
# `hue` takes the value of the attribute `kmeans.labels_`, which is the result of running the k-means algorithm.
# `s` represents the size of the points you want to see in the plot.

ax.grid(False) # This removes the grid from the background.
plt.xlabel('Freedom of choice') # Label x-axis.
plt.ylabel('Score') # Label y-axis.
plt.show()


# In[41]:


# Plot the clusters for the "Score" and "Generosity" variables.

plt.figure(figsize=(12,8))
ax = sns.scatterplot(x=df['Score'], y=df['Life expectancy'], hue=kmeans.labels_, s=100) 
# Here, you're subsetting `X` for the x and y arguments to avoid using their labels. 
# `hue` takes the value of the attribute `kmeans.labels_`, which is the result of running the k-means algorithm.
# `s` represents the size of the points you want to see in the plot.

ax.grid(False) # This removes the grid from the background.
plt.xlabel('Generosity') # Label x-axis.
plt.ylabel('Score') # Label y-axis.
plt.show()


# In[42]:


# Plot the clusters for the "Score" and "Perceptions of corruption" variables.

plt.figure(figsize=(12,8))
ax = sns.scatterplot(x=df['Score'], y=df['Life expectancy'], hue=kmeans.labels_, s=100) 
# Here, you're subsetting `X` for the x and y arguments to avoid using their labels. 
# `hue` takes the value of the attribute `kmeans.labels_`, which is the result of running the k-means algorithm.
# `s` represents the size of the points you want to see in the plot.

ax.grid(False) # This removes the grid from the background.
plt.xlabel('Perceptions of corruption') # Label x-axis.
plt.ylabel('Score') # Label y-axis.
plt.show()


# 8. Discuss how and why the clusters make sense. If they don’t make sense, however, this is also useful insight, as it means you’ll need to explore the data further.
# 
# There are3 distinct clusters as identified by the elbow technique. These clusters apply to each of the variables that are neccessary in explaining respective countries happiness scores. The clusters all makes sense because there is little to no overlap between there are generally well defined paramteres between the clusters allowing for easy identification. For this analysis, the three clusters make sense because they help to identify low, medium and high levels of variable grading allowing analysts to easily spot how these different gradings of vafriable score in terms of happiness. 

# In[44]:


df.loc[df['clusters'] == 2, 'cluster'] = 'black'
df.loc[df['clusters'] == 1, 'cluster'] = 'purple'
df.loc[df['clusters'] == 0, 'cluster'] = 'pink'


# In[46]:


df.groupby('cluster').agg({'GDP per capita':['mean', 'median'], 
                         'Life expectancy':['mean', 'median'], 
                         'Social support':['mean', 'median'],
                           'Perceptions of corruption':['mean', 'median'],
                          'Generosity':['mean', 'median']})


# ANALYSIS: 
# 
# - The black cluster has the best stats across all categories
# 
# - The purple cluster ranks relatviely low comapred to the rest indicating a bigger gap between purple and pink than pink and black. 
# 
# - The exception here, however, relates to the purple category having higher mean values than pink in the fields of "Perceptions of corruption" and "generosity" 
# 
# - Generosity is significantly higher in the black category compared to pink and purple.

# ## Propose what these results could be useful for in future steps of an analytics pipeline

# These results are useful as they provide a clear indication of which variables contribute more to happiness scores, and covnersely which ones are weaker indicators. It also provides parameters that allow for analysis of countries of different scores for more in depth analysis of a particular score class.
# 
# It allows for the formulation of new hypothesis formulation due to the patterns in the variables that have been identified. 

# In[ ]:




