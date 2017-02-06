# Simple Linear Regression
## The machine here is the simple linear regression model
## Learning means that trained the simple linear regression model on the training set
## It also means that the machine learnt the correlation between two variables with the training set


## BEGINNING OF DATA PREPROCESSING ##
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # Taking the Year's of experience data
Y = dataset.iloc[:, -1].values  # Taking the Salary data

# Splitting the dataset into the Training set and Test set
# Set random_state to have a consistent result
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

## END OF DATA PREPROCESSING ##

## Fitting Simple Linear Regression to the training set ##

# Import library scikit learn linear model to import class LinearRegression
# USe simple linear regression model to learn X_train and Y_train
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
#Simple linear regressor learnt through X_train and Y_train

# Predicting new observation with this learnt regressor
# Y_pred predicted value for the dependent variable using X_test
Y_pred = regressor.predict(X_test)


# Visualizing the training set result
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), 'blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show

# Visualizing the test set result
plt.scatter(X_test, Y_test, color = 'yellow')
plt.plot(X_train, regressor.predict(X_train), 'blue') 
# The object regressor has already learnt through training set. 
# So the regression line is the same
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show

a = regressor.predict(1)