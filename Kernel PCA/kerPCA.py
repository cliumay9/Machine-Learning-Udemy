# Kernel PCA
"""

For nonlinear problems. that is, data that is nonlinear seperable


Summary: Preprocess the data > Apply Kernel PCA inside > Apply Linear classifier, Logistic Regression

"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
# prepare the dataset with Matrix X, Y indendent and dependent variables
dataset = pd.read_csv("Social_Network_Ads.csv")

# Matrix the X and Y, independent and dependent variables
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#Feature Scaling (MUST for PCA, LDA)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

## End of Data Preprocessing

### Applying Kernel PCA  ###
"""
Import class, create object of this class (with kernel = rbf - gaussian), 
and apply fit_transform and transform from 
the object on the training set and test set
"""
from sklearn.decomposition import KernelPCA
kerpca = KernelPCA(n_components = 2, kernel = "rbf") # n_components # of dimension
# use None to see which components has the highest explained variance ratio
# with trained pac, .explained_variance_ratio
X_train = kerpca.fit_transform(X_train)
X_test = kerpca.transform(X_test)

# Now X_train and X_test have 2 indepednet variables with highest explained variance ratio

## Fitting Logistic Regressions to the TrainingSet
# Linear = Logistic regresison
# Class is capital letters while function is not
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0) # setting random_state = 0 to provide consistency
classifier.fit(X_train, Y_train)

## Predict Test set result with classifier
y_pred = classifier.predict(X_test)

## Building Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

## Visualizing the Training set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step= 0.01),
                                np.arange(start=X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step= 0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
                                                alpha =.75, cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set== j,0],X_set[Y_set ==j,1],
                c= ListedColormap(('red','green'))(i), label = j)
plt.title('Logistic Regression (Social Network Ads), Ker PCA (Training Set)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

## Visualizing the test set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step= 0.01),
                                np.arange(start=X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step= 0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
                                                alpha =.75, cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set== j,0],X_set[Y_set ==j,1],
                c= ListedColormap(('red','green'))(i), label = j)
plt.title('Logistic Regression(Social Network Ads), Ker PCA (Test Set)')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.show()



