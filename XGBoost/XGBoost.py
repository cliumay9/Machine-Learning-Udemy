# XGBoost
""" Fastest Gradient Boosting"""
# Install xgboost; follow instructions: 
    #http://xgboost.readthedocs.io/en/latest/build.html


### Data Preprocessing ###
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values # Credit score to estimated salary
Y = dataset.iloc[:, -1].values

# Encoding Catogrical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# , OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:,1] = labelencoder_X1.fit_transform(X[:,1]) 
labelencoder_X2 = LabelEncoder()
X[:,2] = labelencoder_X2.fit_transform(X[:,2]) 
### Dummy Variables ###
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
# remove one dummy variable to avoid the trap
X = X[:,1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

### Using XGboost model to fit to the Training set ###
# XGB is a gradient boosting algorithm with trees
# Can use grid search to tune the hyper parameter with XGBClassifier
from xgboost import XGBClassifier
classifier =XGBClassifier(learning_rate =0.1, n_estimators =100, 
                   objective="binary:logistic", gamma=0.01)
#Binary outcome >> binary:logistic objective
classifier.fit(X_train, Y_train)

# Predicting the test set with our xgboost model ###
Y_pred = classifier.predict(X_test)
# Convert Y_pred to binary based on probability by picking a threshold
Y_pred = (Y_pred >0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Apply k-fold cross validation - cross validation applies on trainng set
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
#Obtain 10 accuracies(cv) for each fold
accuracy = accuracies.mean() #mean of those 10 accuracies
std = accuracies.std() #Standard deviation of those 10 accuracies
classifier.apply(X_test)