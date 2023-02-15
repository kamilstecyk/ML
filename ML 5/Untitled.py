


# comment plotting because it causes slow execution !!! 

# we import dataset breast_cancer

from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True) 
#print(data_breast_cancer['DESCR'])


X1 = data_breast_cancer["data"]
y1 = data_breast_cancer["target"]





# we create second dataset   (df) 

import numpy as np
import pandas as pd
size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4 
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')





# we dvide breast_cancer dataset on learning and testings sets

from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1, test_size=0.20, random_state=42)





# we are finding the best depth of decision tree in order to maximize f1

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


# we check for training set


# List of values to try for max_depth:
max_depth_range = list(range(1, 10))
# List to store the accuracy for each value of max_depth:
accuracyTrain = []



for depth in max_depth_range:
    
    dtc = DecisionTreeClassifier(max_depth = depth, 
                             random_state = 42)
    dtc.fit(X_train1, y_train1)
    
    y_predictions = dtc.predict(X_train1)
    
    acc = f1_score(y_train1,y_predictions)
    
    accuracyTrain.append(acc)

print(accuracyTrain) # depth 8 is optimal, because later we have the same resuslts

foundedDepthTrain = 8




dtc = DecisionTreeClassifier(max_depth = foundedDepthTrain, random_state = 42)
dtc.fit(X_train1, y_train1)



    
    


    
    
    




# we check for test set

# List to store the accuracy for each value of max_depth:
accuracyTest = []


for depth in max_depth_range:
    
    dtc = DecisionTreeClassifier(max_depth = depth, 
                             random_state = 42)
    dtc.fit(X_test1, y_test1)
    
    y_predictions = dtc.predict(X_test1)
    
    acc = f1_score(y_test1,y_predictions)
    
    accuracyTest.append(acc)

print(accuracyTest) # depth  is optimal, because later we have the same resuslts

foundedDepthTest = 4


treeDepth = min(foundedDepthTrain,foundedDepthTest)


print(treeDepth)



dtc = DecisionTreeClassifier(max_depth = treeDepth, random_state = 42)
dtc.fit(X_train1, y_train1)





# we prepare data to pickle

from sklearn.metrics import accuracy_score









# we want to display the tree

from sklearn.tree import export_graphviz

f = "dtc.dot"

export_graphviz(dtc,out_file=f,rounded=True,filled=True) 
print(f)
#dtc.dot

import graphviz

graph = graphviz.Source.from_file(f)
graph
str_dot = export_graphviz(dtc,rounded=True,filled=True)

graph = graphviz.Source(str_dot)
graph

from subprocess import check_call
check_call(['dot','-Tpng','dtc.dot','-o','bc.png'])
















