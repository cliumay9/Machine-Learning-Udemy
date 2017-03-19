# Natural Language Processing

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', 
                      quoting =3 ) # Get rid of quotes
# Data Preprocessing
# Cleaning the texts for our simple bag-of-words model
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i]) # Keeping a-z and A-Z
    review =  review.lower()    #make everything to lowercase
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
# Its faster for python to go through words in set(*) than list

# Create the Bag of Words Model
# Just like classification with binary outcomes
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500 ) 
# Obtain unique 1500 most frequent words
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:,1].values

# Build our model with training data
# Usually we use Naive Bayes, Decision tree or Random Forest Classificaiton
# CART, C5.0, Maximum Entropy
### Naive Bayes Model ###
# Split the data into testing and training set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the TrainingSet
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
#Set random state to 0 to provide consistency
classifier.fit(X_train, Y_train)

## Predict Test set result with classifier
y_pred = classifier.predict(X_test)

## Building Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
prec = cm[1][1]/(cm[0][1]+cm[1][1])
recall = cm[1][1]/(cm[1][0]+cm[1][1])
F1 = 2*prec*recall/(prec+recall)
print("Accuracy Rate - NB: ", accuracy)
print("Precsion - NB: ", prec)
print("Recall - NB: ", recall)
print('F1 - NB :', F1)

### End of NB Model ###

### Start of Decision Tree Classificaiton ###
## Fitting Decision Tree Classificaiton to the TrainingSet
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion ="entropy", random_state=0)

classifier.fit(X_train, Y_train)


## Predict Test set result with classifier
y_pred = classifier.predict(X_test)

## Building Confusion Matrix
cm = confusion_matrix(Y_test, y_pred)
cm = confusion_matrix(Y_test, y_pred)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
prec = cm[1][1]/(cm[0][1]+cm[1][1])
recall = cm[1][1]/(cm[1][0]+cm[1][1])
F1 = 2*prec*recall/(prec+recall)
print("Accuracy Rate - Decison Tree Classification : ", accuracy)
print("Precsion - Decison Tree Classification : ", prec)
print("Recall - Decison Tree Classification : ", recall)
print('F1 - Decison Tree Classification :', F1)
### End of Decision Tree Classificaiton

### Start of Random FOrest Classificaiton ###
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 15 , criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


## Predict Test set result with classifier
y_pred = classifier.predict(X_test)

## Building Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
prec = cm[1][1]/(cm[0][1]+cm[1][1])
recall = cm[1][1]/(cm[1][0]+cm[1][1])
F1 = 2*prec*recall/(prec+recall)
print("Accuracy Rate - Random Forest Classification : ", accuracy)
print("Precsion - Random Forest Classification : ", prec)
print("Recall - Random Forest Classification : ", recall)
print('F1 - Random Forest Classification :', F1)
### End of Random Forest Classifcation


