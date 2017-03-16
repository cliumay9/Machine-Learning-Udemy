# SVR
# Remember to install the packages for the library
# Make sure to set the right working directory
setwd("~/Desktop/Ex_Files_DSF_DataMining/Machine-Learning-Udemy/SupportVectorRegression")
# Importing the dataset
dataset = read.csv('PositionSalaries.csv')
dataset = dataset[2:3]

# Fitting the SVR to the dataset
# Create SVR from e1071's SVM's function (Support Vector Machine)
# install.packages('e1071') to install package
library(e1071)
regressor = svm(Salary ~.,
                data = dataset,
                type = 'eps-regression',
                kernel = 'radial')
# Note the highest position level is an outlier so it was ignored by the model.

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the SVR results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Position Level vs Salary (SVR Model)') +
  xlab('Level') +
  ylab('Salary')