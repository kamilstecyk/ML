#!/usr/bin/env python
# coding: utf-8

# In[1]:


# we load our dataset

import pandas as pd
import numpy as np
from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


X = data_breast_cancer["data"].iloc[:, [1,8]]
y = data_breast_cancer["target"]



# In[2]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[3]:


# we define classifiers

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

dtc = DecisionTreeClassifier(random_state=42)
lrc = LogisticRegression(random_state=42)
nbrs = KNeighborsClassifier()


# In[4]:


# we define voting classifier

from sklearn.ensemble import VotingClassifier

voting_clf_hard = VotingClassifier(estimators=[('dt', dtc),('lr', lrc), ('knn', nbrs)],voting='hard')

voting_clf_soft = VotingClassifier(estimators=[('dt', dtc),('lr', lrc), ('knn', nbrs)],voting='soft')


# we train our votings classifiers

voting_clf_hard.fit(X_train,y_train)
voting_clf_soft.fit(X_train,y_train)


# In[5]:


# we check te accuracy

from sklearn.metrics import accuracy_score

# firstly we check accuracies for alone classifiers sequentially

#accuracy_score(y_true, y_pred)

accuracyList = []

for clf in voting_clf_hard.estimators_:
    
    # we predict train set
    y_predTrain = clf.predict(X_train)
    
    # we predict test set
    y_predTest = clf.predict(X_test)
    
    accuracyTrain = accuracy_score(y_train, y_predTrain)
    accuracyTest = accuracy_score(y_test, y_predTest)
    
    accuracyList.append((accuracyTrain,accuracyTest))


for clf in [voting_clf_hard,voting_clf_soft]:
    
    # we predict train set
    y_predTrain = clf.predict(X_train)
    
    # we predict test set
    y_predTest = clf.predict(X_test)
    
    accuracyTrain = accuracy_score(y_train, y_predTrain)
    accuracyTest = accuracy_score(y_test, y_predTest)
    
    accuracyList.append((accuracyTrain,accuracyTest))
    
    
#accuracyList

listOfClassifiers = voting_clf_hard.estimators_ + [voting_clf_hard,voting_clf_soft]
#listOfClassifiers
    


# In[6]:


# we pickling the data

import pickle

with open('acc_vote.pkl', 'wb') as fh:
    pickle.dump(accuracyList, fh)


with open('vote.pkl', 'wb') as fh:
    pickle.dump(listOfClassifiers, fh)

# check unpickling


# In[ ]:


# bagging

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,random_state=42,bootstrap=True)
bag_clf.fit(X_train,y_train)


# In[ ]:


# bagging with 50 % instances

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


bag_clf50 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,random_state=42,max_samples=0.5,bootstrap=True)
bag_clf50.fit(X_train,y_train)


# In[ ]:


# pasting   ( without repetition of samples )

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


pas_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,random_state=42,bootstrap=False)
pas_clf.fit(X_train,y_train)


# In[ ]:


# bagging with 50 % instances

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


pas_clf50 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,random_state=42,max_samples=0.5,bootstrap=False)
pas_clf50.fit(X_train,y_train)

