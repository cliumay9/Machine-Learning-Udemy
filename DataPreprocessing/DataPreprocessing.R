#Data Preprocessing with R

# Importing the dataset
# Note: Make sure you set the correct working directory
# Use ifelse to fix missing data on $Age and $Salary

dataset = read.csv('Data.csv')
dataset$Age = ifelse(is.na(dataset$Age), ave(dataset$Age, FUN = function(x) mean(x,na.rm=TRUE)), dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary), ave(dataset$Salary, FUN = function(x) mean(x,na.rm=TRUE)), dataset$Salary)

# Encoding Categorical data - Country and Purchased
dataset$Country = factor(dataset$Country, levels = c('France', 'Spain','Germany'), labels = c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased, levels = c('Yes', 'No'), labels = c(1, 0))

# Splitting Dataset to Training set and Testing set
# Install and Import library catools to good split
# install.packages('caTools')
library(caTools)

#To provide consistency and reprodcution of the same result #set.seed()
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)  #Split Ratio of training data
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

#Feature Scaling
training_set[, 2:3] = scale(training_set[, 2:3])
testing_set[, 2:3] = scale(testing_set[, 2:3])
