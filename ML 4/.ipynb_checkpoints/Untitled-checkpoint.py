#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn import datasets

# we import 1 set

data_breast_cancer = datasets.load_breast_cancer(as_frame = True)
#print(data_breast_cancer['DESCR'])

type(data_breast_cancer)



X1 = data_breast_cancer["data"].loc[:, ["mean area", "mean smoothness"]]  # area and smoothness
y1 = data_breast_cancer["target"]




# In[25]:


# we import 2 set
import numpy as np

data_iris = datasets.load_iris(as_frame=True)
#print(data_iris['DESCR'])

#print(data_iris)

X2 = data_iris["data"].iloc[:, [2, 3]]
y2 = (data_iris["target"] == 2).astype(np.int8)



# In[8]:


# we divide test into train and test set

from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1, test_size=0.20, random_state=42)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2, test_size=0.20, random_state=42)




# In[10]:


# we build a model for breast_cancer dataset


from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# without scaling

svm_clf = LinearSVC(C=1,loss="hinge",random_state=42)

svm_clf.fit(X_train1, y_train1)


# with scaling

svm_clf_with_scaling = Pipeline([("scaler", StandardScaler()),("linear_svc", LinearSVC(C=1,loss="hinge",random_state=42)),])

svm_clf_with_scaling.fit(X_train1, y_train1)


# In[16]:


# we calculate accuracy for our svm's


# accuracies for classifirer without scaling

y1trainPredictions = svm_clf.predict(X_train1)
accuracy1 = accuracy_score(y_train1,y1trainPredictions)

print(accuracy1)

y1testPredictions = svm_clf.predict(X_test1)
accuracy2 = accuracy_score(y_test1,y1testPredictions)


print(accuracy2)


# accuracies for classifirer with scaling

y1trainPredictions2 = svm_clf_with_scaling.predict(X_train1)
accuracy3 = accuracy_score(y_train1,y1trainPredictions2)

print(accuracy3)

y1testPredictions2 = svm_clf_with_scaling.predict(X_test1)
accuracy4 = accuracy_score(y_test1,y1testPredictions2)


print(accuracy4)




# In[15]:


# we pickling our data

import pickle

accuracyList = [accuracy1,accuracy2,accuracy3,accuracy4]

print(accuracyList)

pickle_out = open("bc_acc.pkl","wb")
pickle.dump(accuracyList, pickle_out)
pickle_out.close()



# In[26]:


# we repeat the same actions for the dataset with irises





# without scaling

svm_clfIris = LinearSVC(C=1,loss="hinge",random_state=42)

svm_clfIris.fit(X_train2, y_train2)


# with scaling

svm_clf_with_scalingIris = Pipeline([("scaler", StandardScaler()),("linear_svc", LinearSVC(C=1,loss="hinge",random_state=42)),])

svm_clf_with_scalingIris.fit(X_train2, y_train2)


# In[27]:


# we calculate accuracies for classifers for second datasets with iris


# accuracies for classifirer without scaling

y2trainPredictionsIris = svm_clfIris.predict(X_train2)
accuracy1Iris = accuracy_score(y_train2,y2trainPredictionsIris)

print(accuracy1Iris)

y2testPredictionsIris = svm_clfIris.predict(X_test2)
accuracy2Iris = accuracy_score(y_test2,y2testPredictionsIris)


print(accuracy2Iris)


# accuracies for classifirer with scaling

y2trainPredictions2Iris = svm_clf_with_scalingIris.predict(X_train2)
accuracy3Iris = accuracy_score(y_train2,y2trainPredictions2Iris)

print(accuracy3Iris)

y2testPredictions2Iris = svm_clf_with_scalingIris.predict(X_test2)
accuracy4Iris = accuracy_score(y_test2,y2testPredictions2Iris)


print(accuracy4Iris)


# In[ ]:




