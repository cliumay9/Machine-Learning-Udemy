# k-means Clustering
setwd("~/Desktop/Ex_Files_DSF_DataMining/Machine-Learning-Udemy/k-means clustering")
dataset = read.csv("Mall_Customers.csv")
X = dataset[4:5]

#Using elbow method to find the optimal kmeans
set.seed(123)
wcss =vector()
for (i in 1:10) wcss[i] = sum(kmeans(X,i)$withinss)
plot(1:10, wcss, type = 'b', main = paste('Cluster of Clients'), 
     xlab= "Number of Clusters", ylab='wcss')
# 5 clusters is the optimal clusters.

# Fitting X into our model kmeans
kmeans = kmeans(X, centers = 5, iter.max=300, nstart=10)

#Visualzing the clusters
library(cluster)
clusplot(X,kmeans$cluster, lines = 0,shade =TRUE, color = TRUE,labels = 2, plotchar = FALSE,span = TRUE, main=paste('Clusters of clients'),xlab ="Annual Income", ylab ="Spending Score")



