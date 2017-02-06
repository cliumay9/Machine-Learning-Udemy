# Simple Linear Regression

## Data Preprocessing ##
# note: remember to set the right working directory
# with setwd('')
# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
## End of Data Processing ##

# Fitting Simple Linear Regression to the training set i.e. building the model - linear model lm
# model build on the training set
regressor = lm(Salary ~ YearsExperience, data = training_set) 
#regressor is trained with the training set

# Predict new observation Test set
y_pred = predict(regressor, newdata = test_set)

# Visualizing the training set and test set results
# Import ggplot2 library if you dont have ggplot2 install
# install.packages('ggplot2')
# Training Set results
library(ggplot2)
ggplot()+
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red')+
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), 
            colour = 'blue')+
  ggtitle(' Salary vs Years of Experience (Training Set)')+
  xlab('Years of Experience')+
  ylab('Salary')

# Test Set results against the regression line
# no need to change training_set to test_set for geom_line the regression line, because our regressor is already trained with the training set
ggplot()+
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red')+
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), 
            colour = 'blue')+
  ggtitle(' Salary vs Years of Experience (Test Set)')+
  xlab('Years of Experience')+
  ylab('Salary')



