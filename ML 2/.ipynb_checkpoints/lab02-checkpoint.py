#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from sklearn.datasets import fetch_openml 
 
mnist = fetch_openml('mnist_784', version=1)


# In[20]:


print((np.array(mnist.data.loc[50]).reshape(28, 28) > 0).astype(int))
    
mnist.target


# In[49]:


import pandas as pd  

# we create dataframes for features and labels

dfFeatures = pd.DataFrame(mnist.data)
dfLabels = pd.DataFrame(mnist.target)


#sorting labeels ascending

y = dfLabels.sort_values("class")

# reindexing feature df



indexes = y.index

"""for i in indexes:
    print(i)"""
    
    
X = dfFeatures.reindex(indexes)



# we have x (features) and y (labels)  


# In[57]:


# we split our datasets

X_train, X_test = X[:56000], X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# checking classes

y_train["class"].unique()


# In[58]:


y_test['class'].unique()


# In[81]:


# we divide our data with scikit-learn function
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( dfFeatures, dfLabels, test_size=0.20, random_state=42)


# we check classes as in previous
y_train["class"].unique()


# In[82]:


y_test["class"].unique()


# In[97]:


# learning

from sklearn.linear_model import SGDClassifier

#print((y_train["class"] == 0))

y_train_0 = (y_train["class"] == 0)
y_test_0 = (y_test["class"] == 0)

print(np.unique(y_train_0))
#print(y_train_0)
#print(len(y_train_0))

sgd_clf = SGDClassifier(random_state=42) 
#sgd_clf.fit(X_train,y_train_0)


# In[ ]:





# In[ ]:




