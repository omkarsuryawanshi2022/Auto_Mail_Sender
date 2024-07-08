from sklearn.datasets import load_iris

iris = load_iris()

print("Features name of iris data set")
print(iris.feature_names)

print("Target name of iris data set")
print(iris.target_names)

print("first 10 element from iris data set")

for i in range(len(iris.target)):
    print("ID:%d,Label %s, Feature :%s"%(i,iris.data[i],iris.target[i]))
    