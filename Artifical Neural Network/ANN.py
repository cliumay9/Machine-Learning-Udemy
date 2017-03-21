# Building ANN on Churn Modeling data.

"""
Install libraries,i.e. Theano, Tensorflow, and Keras
by using pip
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
pip install --upgrade keras

TensorFlow Python3 Mac with Conda Environment
conda create -n tensorflow python=3.5
source activate tensorflow
For CPU only tensorflow:
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py3-none-any.whl
For GPU enabled tensorflow:
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-0.12.1-py3-none-any.whl
THEN:
    pip3 install --ignore-installed --upgrade $TF_BINARY_URL
"""
### Data Preprocessing ###
# In a way ANN is a classifier/regressor
# Now we are building a ANN classifier
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

#Feature Scaling
# We have to feature scale because we dont want one vairable dominating another one
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# No need to feature scale Ys because they return binary

#### Creating the Artificial Neural Network Classifier###

# Import libraries and packages
import keras
# import 2 modules: Sequential Module and Dense Module
from keras.models import Sequential #initialize the Neural Network
from keras.layers import Dense # Create Layers in the ANN

# Initializing the ANN
''' 2 ways to initilize:
1) Defining the sequence of layers 2) Defining a graph
Now we definie the sequence of layers instead
'''
classifier = Sequential()

# Adding input layer and the first hidden layer
# Note: Choose Rectifier acivate function for hidden layers
# Note1: Sigmoid AF for output layer; logistic function
# Add both input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11 )) #add input and hidden layers
# kernel_initializer to initialize the weight close to zero uniformly and use 
# rectifier activation function relu
# Add second new hidden layers
# Dense(*) add a fully connected ANN

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# Since we have 2 hidden layers for our ANN for simplicity now we adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
"""Need to return 1 unit because of the binary dependent variable because we are in the final layer
    for 3 classes, units = 3 and activation function 'softmax'
    a genearlization of the logistic function  that squashes K-dimensional vector 
    to a K-dimensional vector of rea values in the range(0,1) that add up to 1 
    In Probability theory, the out put of softmax can be used to represent a categorical distribution
    The probability distribution over K different Possible outcomes.
    The Gradient log nomalizer of the categorical probability distribution. 
"""
# Compiling the ANN; Applying Stochastic Gradient Descent onto the whole ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
# 2 outcomes Binary_crossentropy
# 3 outcomes categorical_crossentropy

# Fitting the ANN to the training set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100) 
#epoch and batch_size are art


#### Making the prediciton and evalute how well the model is ###

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
# Convert Y_pred to binary based on probability by picking a threshold
Y_pred = (Y_pred >0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# 83.5% Accuracty!!! 

