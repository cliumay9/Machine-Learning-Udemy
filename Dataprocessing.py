# Data Preprocessing
#With Spyder Python3.5

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset
# prepare the dataset with Matrix X, Y indendent and dependent variables
dataset = pd.read_csv("/Users/calvinliu/desktop/Data.csv")

# Matrix the X and Y, independent and dependent variables
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# Missing data: One method is adding mean value for missing data
# Import class from sklearn.preprocessing to handle the missing data 

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy= "mean", axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])