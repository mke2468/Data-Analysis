#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Project: TMDb Movie Data Analysis
# (cleaned from original data on Kaggle)
# 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# <li><a href="#Recommendations">Recommendations</a></li>    
# </ul> 

# <a id='intro'></a>
# ## Introduction
# 
# >>In this project we will address the analysis of the (TMDb) database of 10,000 and include user assessments and wills.
# Specifically, we will be interested in:
# 
# - Does the high budget increase revenues? 
# - In which year was the highest film revenue ?
# - Does high film revenue do get more popular?
#  

# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# 
# ### General Properties

# We will import the packages we need for this analysis.

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# To makes visualization look nice and to make sure that htose visualization pop up in jupyter notebook.

# In[2]:


import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's look at the dataset to understand it's structure.

# In[3]:


df=pd.read_csv('tmdb-movies.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df.describe()


# 
# ### Data Cleaning (Potential problems include "duplicates, missing data")

# * We review the set of data to understand and find and solve problems.

# In[7]:


df.info()


# *  we see if we have a Duplicates problem !

# *  we see if there is any data missing and delete columns that we do not need .

# In[4]:


df.duplicated()


# In[5]:


#To see the number of duplicate rows.
sum(df.duplicated())


# In[6]:


#To drop existing duplicates.
df.drop_duplicates(inplace=True)
#Checks for any remaining duplicates.
sum(df.duplicated())


# In[7]:


# Delete columns that are not related to the subject of study and analysis.
df.drop(['homepage', 'tagline', 'keywords', 'overview', 'production_companies', 'release_date','cast','director'], axis=1, inplace=True)


# In[8]:


df.info()


# In[9]:


# Let's see the change in a visual.
df.head()
df.hist(figsize=(10,12));


# In[12]:


#fill in missing data . 
df[df.imdb_id.isnull()].hist(figsize=(10,12));


# In[10]:


df.fillna(df.mean(), inplace=True)


# In[11]:


df[df.imdb_id.isnull()]
df.dropna(inplace=True)
df.info()


# In[190]:


# look at what rows look like , to make not missing any thing (visual).
df.hist(figsize=(10,12));


# In[12]:


# Check if any of the columns contain null values.
df.isnull().sum().any()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# > **Note**: 'revenue' is my dependent variable and i'll just be exploring those associations among them.
# 
# ### Research Question 1 ( Does the high budget increase revenues?) 

# In[16]:


df.head(1)


# In[17]:


df.info()


# In[44]:


#get count Popular  in dataset
pd.plotting.scatter_matrix(df,figsize=(15,15));


# ###### * To take a quick view of the relationships between the variables (scatter chart, gradually included) for each variable, in order to determine the status of budget and revenues between variables. 

# In[117]:


# get the high budget and revenues (visual) 
df.plot(x='budget', y='revenue', kind='scatter',figsize=(10,10),title='The relationship between revenue and budget');


# * from the plot Above , We can say that the relationship is inverse and not direct and the answer to our question is (no).

# ### Research Question 2  ( In which year was the highest film revenue ? )

# In[14]:


# in which year was the highest film revenue 
df_r=df['release_year'].hist(bins=10)
df_r.set_ylabel('number')
df_r.set_xlabel('revenue')
df_r.set_title('A histogram that depicts the count distribution for revenue');


# ###### *  we can say 2015 is the highest film revenue.

# In[278]:


# Use the Description function to check the previous result. 
df_1=df.release_year.describe()
df_1


# In[104]:


# Pie chart, to Describe the result formally and confirm the numerical result
labels = '25%', '50%', '75%', 'max'
sizes = [ 25, 50, 75, 90]
explode = (0, 0, 0, 0.1)  # only "explode" the 2nd slice (i.e. 'max')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# * It is very clear that the above conclusion is true

# In[30]:


# Validate the result
df.groupby('release_year').revenue.std().plot(kind='pie', figsize=(18,18));


# ###### *  STD shows growing revenue over time.

# * We conclude that the answer to the above question is (2015)

# ### Research Question 3  ( Does high film revenue do get more popular? )

# In[81]:


# Display revenue by popularity
df_popul=df.groupby('revenue')['popularity'].value_counts()
df_popul


# In[170]:


df.revenue.describe()


# In[86]:


# View the min, 25%, 50%, 75%, max popularity values with describe()
df.describe().popularity


# In[44]:


# Bin edges that will be used to "cut" the data into groups
bin_edges=[1.5,1.8,2.2,3.1,33.0]
# Labels for the four popularity level groups
bin_names = ['p_low','p_medium','p_med_high','p_high'] 
# Creates popularity_levels column
df['popularity_levels'] = pd.cut(df['popularity'], bin_edges, labels=bin_names)

# Checks for successful creation of this column
df.head(1)


# In[45]:


# Find the mean quality of each popularity level with groupby
df_pop=df.groupby('popularity_levels').mean().revenue
df_pop


# * To see the result visually

# In[103]:


# Pie chart, to Describe the result formally and confirm the numerical result
labels = 'low popularity', 'medium popularity', 'med_high popularity', 'high popularity'
sizes = [ 1, 2, 3, 5] #Use rounded values
explode = (0, 0, 0, 0.1)  # only "explode" the 2nd slice (i.e. 'low popularity')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# ##### * We find that the largest part is due to high popularity

# In[101]:


# Use query to select each group and get its mean quality
median = df['popularity'].median()
low = df.query('popularity < {}'.format(median))
high = df.query('popularity >= {}'.format(median))

mean_revenue_low = low['revenue'].mean()
mean_revenue_high = high['revenue'].mean()


# In[102]:


# Create a bar chart with proper labels
locations = [1, 2]
heights = [mean_revenue_low, mean_revenue_high]
labels = ['Low', 'High']
plt.bar(locations, heights, tick_label=labels)
plt.title('Average revenue Ratings by popularity')
plt.xlabel('popularity Content')
plt.ylabel('Average revenue Rating');


# * We can say that higher revenues reap greater popularity  

# <a id='conclusions'></a>
# ## Conclusions
# 
# ####  Questions to analyze the TMDb Movie dataset :
# - Does the high budget increase revenues? 
# - In which year was the highest film revenue ?
# - Does high film revenue do get more popular?
# 
# #### The answer to the questions :
# * from the plot Above , We can say that the relationship is inverse and not direct and the answer to our question is (no).
# * We conclude that the answer to the above question is (2015).
# * We can say that higher revenues reap greater popularity.
# 
# #### Sharing of events :
# - The number of samples is suitable for a simple to moderate analysis of the level of change in the entire site or the possibilities provided at the site.
# - Duplicate column was deleted and did not result in any malfunctioning of data structure or analysis results.
# - Columns that had no effect on the analysis or results presented were omitted.
# - The text data was not analyzed in a detailed and integrated way because the variables selected for the analysis did not include variables (columns) text.
# 
# ***In conclusion, we have data indicating that revenues are an important factor and fall in the classification and popularity of films and that the selection of revenues as the focus of analysis was a valid decision and works well.
# 
# ### Note:
# N/A (I only used the site that is in the Udasti site).

# <a id='Recommendations'></a>
# ## Recommendations
# from my point of view :
# - It is preferable to put each type of film in a different column or a different division method is easy to deal with, as well as a lot of text columns that have the same problem.
# - If the names of the columns were shortened and put a list of these abbreviations in order to be easier to deal with.
# - A defect was observed in the cast column, but it was not treated before it was stored in the data set.
# - The difference between columns (id ,imdb_id ) is not clarified.
# 
# **Perform a detailed analysis of the descriptive data (strings) in this dataset in order to obtain an analysis of the entire dataset presented here.

# In[105]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




