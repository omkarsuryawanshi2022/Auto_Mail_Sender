# import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the iris data set with pandas

dataset = pd.read_csv('iris.csv')
x = dataset.iloc[:,[0,1,2,3]].values

#finding the optimum number of clusters for  k -means classification
from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    Kmeans = KMeans(n_clusters = i, init = 'k-means++'
    ,max_iter = 300, n_init = 10, random_state = 0)
    Kmeans.fit(x)
    wcss.append(Kmeans.inertia_)

# plotting the result onto a line graph, allowing us to observe 'the elbow'
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('number of clusters')
plt.ylabel('WCSS') # within cluster sum of squares
plt.show()

#Applaying kmeans to the dataset/ creating the kmeans classifier
kmeans = KMeans(n_clusters = 3 ,init = 'k-means++', max_iter = 300, n_init =
10,random_state = 0)
y_kmeans = kmeans.fit_predict(x)

# visualising the cluster
plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1], s = 100, c = 'red',label
= 'Iris-sentosa')

plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1], s = 100, c = 'blue', label
= 'Iris-versicolor')

plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1], s = 100, c = 'green',
label = 'Iris-virginica')

# plotting the centroid  of the clusters

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100,
c = 'yellow',label = 'Centroid')

plt.legend()

plt.show()



