# Decision Tree Classification
setwd("~/Desktop/Ex_Files_DSF_DataMining/Machine-Learning-Udemy/DecisionTreeClassification")

dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[,3:5]

# Encode the target feature as factor 
dataset$Purchased = factor(dataset$Purchased, levels = c(0,1))
# dataset$Gender = factor(dataset$Gender, levels = c('Male','Female'), labels = c(0,1))
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

## Creating Decision Tree Classification classifier
# install.packages('rpart')
library(rpart)
classifier = rpart(Purchased~. , data=training_set)

y_pred = predict(classifier, newdata=test_set[-3],type = 'class')
#Building Confusion matrix

cm= table(test_set[,3],y_pred)

# Visualize
# install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set
X1= seq(min(set[,1])-1, max(set[,1])+1, by =0.01)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by =0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set)=c('Age','EstimatedSalary')
y_grid = predict(classifier, newdata= grid_set, type = 'class')
plot(set[,-3],
     main = "Decision Tree Classification (Training Set)",
     xlab = 'Age', ylab='Estimated Salary',
     xlim = range(X1), ylim=range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add= TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid==1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[,3]==1, 'green4','red3'))

# Visualizing the test set result
library(ElemStatLearn)
set = test_set
X1= seq(min(set[,1])-1, max(set[,1])+1, by =0.01)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by =0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set)=c('Age','EstimatedSalary')
y_grid = predict(classifier, type = 'class', newdata= grid_set)
plot(set[,-3],
     main = "Decision Tree Classification (Test Set)",
     xlab = 'Age', ylab='Estimated Salary',
     xlim = range(X1), ylim=range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add= TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid==1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[,3]==1, 'green4','red3'))

# Plot the decision tree. DONT FEATURE SCALE
plot(classifier)
text(classifier)