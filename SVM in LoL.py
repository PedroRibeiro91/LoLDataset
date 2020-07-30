#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[5]:


full_data = pd.read_csv(r'\Users\Pedro\Documents\high_diamond_ranked_10min.csv')


# In[12]:


X = full_data.drop(['gameId', 'blueWins'], axis=1)
Y = full_data['blueWins'].values.reshape(-1,1)


# In[15]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.9, random_state = 69)


# In[16]:


from sklearn import svm


# In[17]:


svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(X_train,Y_train)


# In[18]:


Y_pred = svm_clf.predict(X_test)


# In[19]:


from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))


# In[20]:


# lets take a look at our models precision and recall

print("Precision:",metrics.precision_score(Y_test, Y_pred))

print("Recall:",metrics.recall_score(Y_test, Y_pred))


# In[ ]:


# the svm model is barely acceptable
# but precision and recall stay a little behind from what its desired
# some improvements to the svm maybe brought by trying another kernel, try different train and test sizes
# But as it is a slow model especially on low specs rigs these experiments will be left for another time

