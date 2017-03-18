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



