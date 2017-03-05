# K-mean Clustering

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## import the mall data set with pandas
dataset = pd.read_csv('./Mall_Customers.csv')
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
kmeans = KMeans(n_clusters = 5, init= 'k-means++', max_iter=300,
                n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the clusters; 2 dimensional clustering

plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0, 1], s =100, c='red', label ='Careful Client')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1, 1], s =100, c='blue', label ='Standard Client')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2, 1], s =100, c='green', label =' Target Client *')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3, 1], s =100, c='cyan', label ='Careless Client')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4, 1], s =100, c='magenta', label ='cluster5: Sensible Clients')
# Plotting the centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300, c='yellow', label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income(in $k)')
plt.ylabel('Spending Score(1-100')
plt.legend()
plt.show()