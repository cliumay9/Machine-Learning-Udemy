# K-mean Clustering

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## import the mall data set with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

### Choose number of clusters with elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)



plt.plot(range(1,11), wcss)
plt.title("The Elbow Method, Number of Customer vs WCSS (Mall Customers:Annual Income, Buying buyer)")
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.show()
# Right number of clusters is found, k =5


# Applying k-means to the model
