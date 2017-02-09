# Multiple Linear Regression

## Begin Data Preprocessing ##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Make sure we are in the correct wd

# y, the profit is the most interested variable for start_up
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Encoding Categorical Data with sklearn.preprocessing's Label Encoder
# However, if we encode with label the dataset itself implies that
# one state value is greater than the other ones
# THus, we use dummy variables as switches. In order to do this, 
# We employ OneHotEncoder from sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()


# Avoiding the Dummy Variable Trap, aka counting more variables which implies one variable related to other by just the category
#Dont take the first column the first dummy variable
# For some libraries, one need to remove the dummy variable manually

X = X[:,1:] 

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

## ENd of Data Preprocessing ##

# Fitting Multiple Linear Regression to the Training set
# regressor is our object in the linear_model class
# fit the object to our regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results with the regressor(The Mult Linear Regression Model)
y_pred = regressor.predict(X_test) 

# Optimize Mult Linear Regression with Backward Elimination
# Remove lower statistic signficant variables
# Import statsmodel.formula.api as sm

import statsmodels.formula.api as sm
# Need to add column of 1's to take care of regression model equaiton
# i.e. Y= b0*X0+b1*X1+b2*X2+...
# axis is 1 because we wanted to add a COLUMN. if one needs to add a row axis =0

X = np.append(arr= np.ones((50,1)).astype(int), values = X, axis=1)

# Start backwardelimination After adding one's
# Create X_opt matrix for the optimal matrix for high impact independent variables
X_opt = X[:,[0,1,2,3,4,5]]
# Let's fit the model to the X_opt by creating new regressor from statsmodel
# Fitting the full model with all possible predictors
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() 
# Note: Ordinary Least Square regressor is created from sm library
# Showed a table with how statistical signficant each variable is.
regressor_OLS.summary() 

# Remove X2, that is dummy variable for New York X[:,2]
# Because X2 P value is 99% >> 5% - our signficant level
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary() 

# Remove X1  (X[:,1]), that is the dummy variable for Florida, because P-value is 94%>> SL=5%
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary() 

# Remvove X2, X[:,4], that is the administration cost
# because its p-value is 0.602>0.05(our sl)
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary() 

# Remove X[:,5] because 0.06>0.05 our sl.
# There are ways to improve our model that will change our signficance level
# But for this model we follow strictly with backwd elimination

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary() 

# Only R& D spent is left
# This example happends to be a linear regression




