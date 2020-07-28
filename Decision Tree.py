#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


full_data = pd.read_csv(r'\Users\Pedro\Documents\high_diamond_ranked_10min.csv')

X = full_data.drop(['gameId', 'blueWins'], axis=1)
Y = full_data['blueWins'].values.reshape(-1,1)


# In[4]:


# we know that theres as many blueWins = 1 as blueWins = 0, because this isnt the first time with this dataset


# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.9, random_state = 69)


# In[6]:


from sklearn.tree import DecisionTreeClassifier


# In[7]:


tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, Y_train)


# In[15]:


Y_pred = tree_clf.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
# percentage of correct predictions
perc_correct = np.diag(cm).sum()/len(X_test)
perc_correct


# In[19]:


# our tree predicted correctly 64% of the test
# lets take a look at the tree
from sklearn.tree import plot_tree
plot_tree(tree_clf, max_depth=3, filled=True)


# In[20]:


# now its time to try and improve our tree
# with something called bagging

from sklearn.ensemble import BaggingClassifier
bag_clf = BaggingClassifier()
bag_clf.fit(X_train,Y_train.ravel())


# In[21]:


Ybag_pred = bag_clf.predict(X_test)
cm_bag = confusion_matrix(Y_test, Ybag_pred)
# percentage of correct predictions
perc_correct_bag = np.diag(cm_bag).sum()/len(X_test)
perc_correct_bag


# In[23]:


# the bagging corrector improve significantly our classification from 0.64 to 0.73, aproximately
# now lets try a random forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(30) # we have to define the number of trees. As this is a very large dataset we will use 30
rf_clf.fit(X_train,Y_train.ravel())


# In[24]:


Yrf_pred = rf_clf.predict(X_test)
cm_rf = confusion_matrix(Y_test, Yrf_pred)
# percentage of correct predictions
perc_correct_rf = np.diag(cm_rf).sum()/len(X_test)
perc_correct_rf


# In[ ]:


# conclusions
# We started with a decision tree which predicted correctly 64% of the test
# then tried a baggings classifier that predicted correctly 73%
# and finally our random forest had the better performance by predicting 75% 
# so for prediction purposes a random forest is the best method out of these 3 to make prediction over this dataset

