# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/calvinliu/Desktop/Ex_Files_DSF_DataMining/Machine-Learning-Udemy/Random Forest Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_Y = StandardScaler()
Y_train = sc_Y.fit_transform(Y_train)"""

# Fitting the Random Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
# n_estimators # of trees 
# Set random_state = 0 to have consistent result
regressor.fit(X,Y)

# Predicting a new result
Y_pred = regressor.predict(6.5)

# Visualising the Random Forest Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Salary vs Position (Random Forest Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()