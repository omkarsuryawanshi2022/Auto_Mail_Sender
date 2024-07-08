import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

print("feature names of iris data set")
print(iris.feature_names)

print("Terget names of iris data set")
print(iris.target_names)

#Indices of removel element
test_index = [1,51,101]

#Traning data with removed elements
train_target = np.delete(iris.target,test_index)
train_data=np.delete(iris.data,test_index,axis=0)

#Testing data for testing on tranning data

test_target=iris.target[test_index]
test_data=iris.data[test_index]

#from decision tree classifier
classifier=tree.DecisionTreeClassifier()

#applay traning data from tree
classifier.fit(train_data,train_target)

print("value that we removed for testing")
print(test_target)

print("Result of testing")
print(classifier.predict(test_data))
