#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This notebook runs on MachineLearningCourse enviroment (for libraries version)
#numpy 1.21.5
#pandas 1.4.1
#matplotlib 3.5.1
#sci-kit learn 1.0.2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


# In[2]:


dataset= pd.read_csv("HousePrices.csv")
print(dataset.info())  
#Conclusion: 414 entries and there isn't a null value to be imputed


# In[3]:


print(dataset.head())


# In[4]:


# some information about the values in transaction date
print(dataset['transaction date'].value_counts())
#Conclusion:
#transaction date (only object-type column in dataset) doesnt have a dominating category so i can't split the values 
#into 2 columns :(1) the dominating category and (2)others to be dummied after 


# In[5]:


plt.scatter(dataset['transaction date'],dataset['house price of unit area'])
plt.xlabel('date', fontsize = 10)
plt.ylabel('price', fontsize = 10)
plt.show()
#Conclusion: it will be really hard to fit a staright line through this particular feature because they are making vertical 
#lines, if a straight line passes through them its slope will be = infinity -> not possible -> need to find workaround
#Conclusion 2: there is a point that maybe an outlier (117.5) -> need to check outliers in general


# In[6]:


print(pd.get_dummies(dataset['transaction date']))
#Conclusion: good approach but 12 dummy variables and this is a univariable linear regression model


# In[7]:


#Taking only the year part as the transaction date
for row in range(len(dataset['transaction date'])):
        year =  int(dataset['transaction date'][row][:4])
        dataset['transaction year']= year
dataset = dataset.drop('transaction date',axis = 1)
print(dataset['transaction year'])


# In[8]:


print(dataset)


# In[9]:


#Finding outliers using interquartile range 
def get_outliers_boundaries(feature):
    first_quartile = np.percentile(feature,25)
    third_quartile = np.percentile(feature,75)
    interquartile_range = third_quartile - first_quartile
    min_num = first_quartile - 1.5 * interquartile_range
    max_num = third_quartile + 1.5 * interquartile_range
    #in this problem, I cant have any negative lower boundary
    if min_num < 0:
        min_num = 0
    return min_num,max_num


# In[10]:


print(dataset.columns[-2])


# In[11]:


print(dataset.shape)
print(dataset[dataset.columns[-2]].max())


# In[12]:


min_num,max_num = get_outliers_boundaries(dataset[dataset.columns[-2]])
print(min_num)
print(max_num)
for i in range(len(dataset[dataset.columns[-2]])):
    if  dataset[dataset.columns[-2]][i]< min_num or dataset[dataset.columns[-2]][i] > max_num:
        dataset = dataset.drop(i,axis=0)


# In[13]:


print(dataset.shape)
print(dataset[dataset.columns[-2]].max())


# In[14]:


#Normalizing values using sigmoid function as it ranges from 0->1
#Conclusion : in latitude and longitude, after normalizing the value they all become (1.0) making a vertical line
#so I will try to scale them 
featureScaler = MinMaxScaler()
to_scale_list = ['distance to the nearest MRT station','latitude','longitude']
for feature in to_scale_list:
    dataset[feature]= featureScaler.fit_transform(np.array(dataset[feature]).reshape(dataset.shape[0],1))


# In[15]:


print(dataset)


# In[16]:


def hypothesis(x,y):
    feature = np.array(dataset[x]).reshape(dataset[x].shape[0],1)
    label = np.array(dataset[y]).reshape(dataset[y].shape[0],1)
    model = LinearRegression()
    model.fit(feature,label)
    predictions = model.predict(feature)
    MSE = metrics.mean_squared_error(label,predictions)
    plt.scatter(dataset[x],dataset[y])
    plt.xlabel(x)
    plt.ylabel(dataset.columns[-2])
    plt.plot(dataset[x],predictions,color='red',linewidth = 4)
    plt.show()
    return MSE


# In[17]:


#the line is not obvious because the difference between the x axis points is very large
MSE_transaction_date = hypothesis(dataset.columns[-1],dataset.columns[-2])
print(MSE_transaction_date)


# In[18]:


MSE_house_age = hypothesis(dataset.columns[0],dataset.columns[-2])
print(MSE_house_age)


# In[19]:


MSE_distance_to_MRT_station = hypothesis(dataset.columns[1],dataset.columns[-2])
print(MSE_distance_to_MRT_station)


# In[20]:


MSE_num_convenience_stores = hypothesis(dataset.columns[2],dataset.columns[-2])
print(MSE_num_convenience_stores)


# In[21]:


MSE_latitude = hypothesis(dataset.columns[3],dataset.columns[-2])
print(MSE_latitude)


# In[22]:


MSE_longitude = hypothesis(dataset.columns[4],dataset.columns[-2])
print(MSE_longitude)


# In[23]:


#Seeing the diffrerence between processed and unprocessed data
unprocessed_dataset=pd.read_csv("assignment1_dataset.csv")
print(unprocessed_dataset)


# In[24]:


def unprocessed_hypothesis(x,y):
    feature = np.array(unprocessed_dataset[x]).reshape(unprocessed_dataset[x].shape[0],1)
    label = np.array(unprocessed_dataset[y]).reshape(unprocessed_dataset[y].shape[0],1)
    model = LinearRegression()
    model.fit(feature,label)
    predictions = model.predict(feature)
    MSE = metrics.mean_squared_error(label,predictions)
    plt.scatter(unprocessed_dataset[x],unprocessed_dataset[y])
    plt.xlabel(x)
    plt.ylabel(unprocessed_dataset.columns[-2])
    plt.plot(unprocessed_dataset[x],predictions,color='red',linewidth = 4)
    plt.show()
    return MSE


# In[25]:


for row in range(len(unprocessed_dataset['transaction date'])):
        year =  int(unprocessed_dataset['transaction date'][row][:4])
        unprocessed_dataset['transaction year']= year
unprocessed_dataset = unprocessed_dataset.drop('transaction date',axis = 1)
print(unprocessed_dataset['transaction year'])


# In[26]:


print(unprocessed_dataset)


# In[27]:


MSE_transaction_date_unprocessed = unprocessed_hypothesis(unprocessed_dataset.columns[-1],unprocessed_dataset.columns[-2])
print(MSE_transaction_date_unprocessed)


# In[28]:


MSE_house_age_unprocessed = unprocessed_hypothesis(unprocessed_dataset.columns[0],unprocessed_dataset.columns[-2])
print(MSE_house_age_unprocessed)


# In[29]:


MSE_distance_to_MRT_station_unprocessed= unprocessed_hypothesis(unprocessed_dataset.columns[1],unprocessed_dataset.columns[-2])
print(MSE_distance_to_MRT_station_unprocessed)


# In[30]:


MSE_num_convenience_stores_unprocessed = unprocessed_hypothesis(unprocessed_dataset.columns[2],unprocessed_dataset.columns[-2])
print(MSE_num_convenience_stores_unprocessed)


# In[31]:


MSE_latitude_unprocessed= unprocessed_hypothesis(unprocessed_dataset.columns[3],unprocessed_dataset.columns[-2])
print(MSE_latitude_unprocessed)


# In[32]:


MSE_longitude_unprocessed= unprocessed_hypothesis(unprocessed_dataset.columns[4],unprocessed_dataset.columns[-2])
print(MSE_longitude_unprocessed)

