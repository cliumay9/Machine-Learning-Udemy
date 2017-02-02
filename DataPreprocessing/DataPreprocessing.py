# Data Preprocessing
# With Spyder Python3.5

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
# prepare the dataset with Matrix X, Y indendent and dependent variables
dataset = pd.read_csv("Data.csv")

# Matrix the X and Y, independent and dependent variables
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# Missing data: One method is adding mean value for missing data
# Import library sklearn.preprocessing to handle the missing data 
# Create object(imputer)with class Imputer

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy= "mean", axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

# Encoding Catogrical Data Country and Purchased
# From library sklearn.preprocessing to import class labelEncoder
# From library sklearn.preprocessing to import class OneHotEncoder to excecute Dummy Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0]) #The contry names are encoded

# However, note that if we categorize countries with different numbers
# It implies that one country value is smaller than the other one's and vice versa
# Dummy Encoding, i.e. each column represents a country
# It's different for 'Purchased' Category because ML knows it is a depdent variable
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Splitting data set to Training data and Testing data
# Import library sklearn.cross_validation import class train_test_split
# Fixed random_state in order to for consistent reproduction
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#Feature Scaling
# Import Library sklearn.preprocessing to import class StandardScaler
# Note: we fit_transform training data but transform test data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

