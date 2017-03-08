# Hierarchical Clustering
setwd("~/Desktop/Ex_Files_DSF_DataMining/Machine-Learning-Udemy/Hierarchical Clustering")

# Import dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

# Use Dendrogram to find the optimal number of clustering
# ward.D minizimizing the variance within clusters
dendrogram = hclust(dist(X, method ='euclidean'), method = 'ward.D')
plot(dendrogram, 
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean Distances')
# finding the largest vertical distance w/o crossing the horzontal distance
# 5 is the optimal number

# Fittng HC model into the dataset
hc = hclust(dist(X, method ='euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5) # Cutting the dendrogram tree , cutree

# Visualizing the dataset
library(cluster)
clusplot(X,y_hc, lines = 0,shade =TRUE, color = TRUE,labels = 2, plotchar = FALSE,span = TRUE, main=paste('Clusters of clients'),xlab ="Annual Income", ylab ="Spending Score")

