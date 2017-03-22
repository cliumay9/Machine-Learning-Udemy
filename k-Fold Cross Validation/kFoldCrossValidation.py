# k-Fold Cross Validation

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
# prepare the dataset with Matrix X, Y indendent and dependent variables
dataset = pd.read_csv("Social_Network_Ads.csv")

# Matrix the X and Y, independent and dependent variables
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
## End of Data Preprocessing

## Fitting SVM to the TrainingSet
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0) #Set random state to 0 to provide consistency
classifier.fit(X_train, Y_train)

### Much better way to evaluate a model, other than Confusion matrix ###
# Apply k-fold cross validation - cross validation applies on trainng set
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
accuracy = accuracies.mean()
std = accuracies.std()
# Accuracies vector composed of 10 elements from those 10 folds give you 10 accuracies
# cv = # of folds
# for faster performance, set n_jobs to -1 to use all CPUS

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
plt.title('Kernel SVM (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
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
plt.title('Kernel SVM (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()