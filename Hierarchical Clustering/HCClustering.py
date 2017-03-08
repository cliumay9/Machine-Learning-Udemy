# Hierarchical Clustering

#import the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset with pd
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# Finding optimal number of clustering with dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))
# Ward minimizing the variance within each clusters
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('L2 Distance, Euclidean Distance')
plt.show()

# To find the optimal clustering, we look for the largest vertical distance without crossing any horizontal lines
# 5 Clusters

# Fit HC model to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters =5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#Visualizng the clusters
plt.scatter(X[y_hc==0,0], X[y_hc==0, 1], s =100, c='red', label ='Careful Client')
plt.scatter(X[y_hc==1,0], X[y_hc==1, 1], s =100, c='blue', label ='Standard Client')
plt.scatter(X[y_hc==2,0], X[y_hc==2, 1], s =100, c='green', label =' Target Client *')
plt.scatter(X[y_hc==3,0], X[y_hc==3, 1], s =100, c='cyan', label ='Careless Client')
plt.scatter(X[y_hc==4,0], X[y_hc==4, 1], s =100, c='magenta', label ='Sensible Clients')
# Plotting the centroids

plt.title('Clusters of clients')
plt.xlabel('Annual Income(in $k)')
plt.ylabel('Spending Score(1-100')
plt.legend()
plt.show()

