# Polynomial Regression
# The machine is the polynomial regression model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## Beginning of Data Preprocessing ##
# Import the dataset
# prepare the dataset with Matrix X, Y indendent and dependent variables
dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, [1]].values 
# We dont need X[:,0]; I can also use 1:2 instead of [1] to make sure X is a matrix
Y = dataset.iloc[:,-1].values 

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)


##  Fitting Polynomial Regression to the data set ##
# Import polynomial class, PolynomialFeatures.(simply add polynomial terms)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X) 
#Fit our object to X then transform X to X_poly
# It automatically add the column of 1's to include the X0
# Fit X_poly into the linear regressor, lin_reg_2.
# X_poly (consist of 2 independent varaibles X, X^2) because of degree = 2
# WE CAN IMPROVE THIS MODEL BY ADDING THE DEGREE
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualizing the Linear Regression Results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), 'blue') #USe lin_reg.predict
plt.title('Position level VS Salary (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
#Visualizing the Poly Reg Ressults
X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid)),1) #More Continuous Curve
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), 'blue') #USe lin_reg.predict
plt.title('Position level VS Salary (Poly Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)


# Predict a new result with Poly Reg
lin_reg_2.predict(poly_reg.fit_transform(6.5))
