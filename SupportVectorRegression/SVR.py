# Support Vector Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Beginning of Data Preprocessing ##
# Import the dataset
# prepare the dataset with Matrix X, Y indendent and dependent variables
dataset = pd.read_csv("PositionSalaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, -1].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_scaled = sc_X.fit_transform(X) #fit and transform
Y_scaled = sc_Y.fit_transform(Y) #fit and transform
# Fitting SVR into the dataset
# Import SVR class from sklearn.svm {svm stands for support vector machine}
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #rbf = radial basis function (Gausian Kernel)
regressor.fit(X_scaled, Y_scaled) 
#SVR got rid of outliers, that is the level 10 points

# Predicting a new result

Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array(6.5)))) #just transform 6.5 to the scale


# Visualising the SVR results
X_grid = np.arange(min(X), max(X), 0.001) #Make the graph look smoother
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, sc_Y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Position Level vs Salary (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()