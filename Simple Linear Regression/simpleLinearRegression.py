# Simple Linear Regression

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

## FItting Simple Linear Regression to the training set ##

# Import library scikit learn linear model to import class LinearRegression
# USe simple linear regression model to learn X_train and Y_train
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


