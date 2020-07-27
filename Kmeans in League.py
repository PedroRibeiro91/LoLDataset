#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[81]:


full_data = pd.read_csv(r'\Users\Pedro\Documents\high_diamond_ranked_10min.csv')


# In[82]:


# full_data.columns


# In[83]:


X = np.array(full_data.drop(['gameId', 'blueWins'],1).astype(float))
Y = np.array(full_data['blueWins'])


# In[84]:


kmeans = KMeans(n_clusters=2) # 2 clusters: blueWins = 1 v blueWins = 0
kmeans.fit(X)


# In[85]:


# model quality
correct = 0
for i in range(len(X)):
    predict = np.array(X[i].astype(float))
    predict = predict.reshape(-1, len(predict))
    prediction = kmeans.predict(predict)
    if prediction[0] == Y[i]:
        correct += 1

print(correct/len(X))


# In[86]:


# only 0.27 correct predictions 
# it's only natural because we used the data as it is
# We have to two options


# In[87]:


# 1 - play around with the parameters, in our kmeans method, such as max_iter and n_jobs 


# In[88]:


# we will try one kmeans with max_iter = 600
kmeans600 = KMeans(n_clusters=2, max_iter=600)
kmeans600.fit(X)


# In[89]:


# model quality
correct_600 = 0
for i in range(len(X)):
    predict_600 = np.array(X[i].astype(float))
    predict_600 = predict_600.reshape(-1, len(predict_600))
    prediction_600 = kmeans600.predict(predict_600)
    if prediction_600[0] == Y[i]:
        correct_600 += 1

print(correct_600/len(X))


# In[90]:


# we can see that for 600 iterations of Kmeans, we have a massive improvement from 0.27 up to 0.72
# but there will be occasions where even with parameter tweaking, the quality of model won't improve at least significantly
# in such case we need
# 2 -  rescale our data


# In[91]:


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[92]:


kmeans.fit(X_scaled)


# In[93]:


correct_scaled = 0
for i in range(len(X)):
    predict_scaled = np.array(X[i].astype(float))
    predict_scaled = predict_scaled.reshape(-1, len(predict_scaled))
    prediction_scaled = kmeans.predict(predict_scaled)
    if prediction_scaled[0] == Y[i]:
        correct_scaled += 1

print(correct_scaled/len(X))


# In[ ]:


# same a similar result was obtained as in the first try with this method
# we can use the 600 interations model and check it improves 


# In[94]:


kmeans600.fit(X_scaled)


# In[96]:


correct_scaled600 = 0
for i in range(len(X)):
    predict_scaled600 = np.array(X[i].astype(float))
    predict_scaled600 = predict_scaled600.reshape(-1, len(predict_scaled600))
    prediction_scaled600 = kmeans600.predict(predict_scaled600)
    if prediction_scaled600[0] == Y[i]:
        correct_scaled600 += 1

print(correct_scaled600/len(X))


# In[64]:


# again similar result as before
# in order to try and make the model betterwe rescaled the date, but it seems that for this dataset rescaling doesnt make 
# that much of a difference and tweaking max iteraction provides a better result in term of the quality of the clusters


# In[ ]:




