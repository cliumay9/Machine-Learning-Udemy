# Multiple Linear Regression

# Note: remember to set the correct Working Directory

## Beginning of Data Preprocessing ##
# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding Categorical data - Country and Purchased
dataset$State = factor(dataset$State, levels = c('New York', 'California', 'Florida'), labels = c(1, 2,3))

library(caTools) #Created dummy varible automatically and avoided the trap
set.seed(123) #consistency and reprodcution 
split = sample.split(dataset$Profit, SplitRatio = 0.8)  #Split Ratio of training data
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)
## End of Data Preprocessing ##

## Fit Multiple Linear Regression model to the Training set
# Use . to include all independent variables
regressor = lm(Profit ~ . , data = training_set)
summary(regressor)

# Predicting the test set results
y_pred = predict(regressor, newdata = testing_set)
####
# Building the optimal model with Backward Elimination
# Remove independent varaibles that has SL less than 0.05
regressor = lm(Profit ~ R.D.Spend + Administration + Marketing.Spend + State , data = training_set)
summary(regressor)
regressor = lm(Profit ~ R.D.Spend + Administration + Marketing.Spend , data = training_set)
summary(regressor)
regressor = lm(Profit ~ R.D.Spend + Marketing.Spend , data = training_set)
summary(regressor)
regressor = lm(Profit ~ R.D.Spend, data = training_set)
summary(regressor)

