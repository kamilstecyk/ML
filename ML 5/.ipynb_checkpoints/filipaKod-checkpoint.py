#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
X = data_breast_cancer["data"]
y = data_breast_cancer["target"]


# In[6]:


import numpy as np
from sklearn.model_selection import train_test_split

X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


from sklearn.tree import DecisionTreeClassifier

tree_clf_bc = DecisionTreeClassifier(max_depth=3,random_state=42)

tree_clf_bc.fit(X_train_bc,y_train_bc)

y_pred_train_bc = tree_clf_bc.predict(X_train_bc)
y_pred_test_bc = tree_clf_bc.predict(X_test_bc)


# In[8]:


from sklearn.metrics import f1_score

f1_score_train = f1_score(y_train_bc,y_pred_train_bc)
print(f1_score_train)

f1_score_test = f1_score(y_test_bc,y_pred_test_bc)
print(f1_score_test)


# In[17]:


from sklearn.tree import export_graphviz

f = "breast_cancer_tree.dot"

export_graphviz(
        tree_clf_bc,
        out_file=f,
        rounded=True,
        filled=True
)
print(f)


# In[18]:


import graphviz

graph = graphviz.Source.from_file(f)


# In[25]:


from sklearn.metrics import accuracy_score

accuracy_train = accuracy_score(y_train_bc,y_pred_train_bc)
accuracy_test = accuracy_score(y_test_bc,y_pred_test_bc)

print(accuracy_train)
print(accuracy_test)

bc_list = [3,f1_score_train,f1_score_test,accuracy_train,accuracy_test]


# In[21]:


import numpy as np
import pandas as pd
size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4 
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')


# In[22]:


from sklearn.tree import DecisionTreeRegressor
X_train_df, X_test_df,y_train_df,y_test_df = train_test_split(df[['x']], df[['y']], test_size=0.2, random_state=42)

regr_bc = DecisionTreeRegressor(max_depth = 4, random_state = 42)

regr_bc.fit(X_train_df,y_train_df)


# In[23]:


from sklearn.metrics import mean_squared_error

y_pred_train_df = regr_bc.predict(X_train_df)
y_pred_test_df = regr_bc.predict(X_test_df)

MSE_df_train = mean_squared_error(y_train_df,y_pred_train_df)
MSE_df_test = mean_squared_error(y_test_df,y_pred_test_df)

print(MSE_df_train)
print(MSE_df_test)


# In[27]:


import pickle

file = open('bc.png', 'wb')
pickle.dump(graph, file)

file = open('f1acc_tree.pkl', 'wb')
pickle.dump(bc_list, file)


# In[29]:


#rysowanie drzewa

f_ = "df.dot"

export_graphviz(
        regr_bc,
        out_file=f_,
        rounded=True,
        filled=True
)
print(f_)

graph_ = graphviz.Source.from_file(f_)


# In[40]:


import matplotlib.pyplot as plt
plt.scatter(X_train_df,y_train_df)
plt.scatter(X_test_df,y_pred_test_df,color="pink")


# In[ ]:





# In[ ]:




