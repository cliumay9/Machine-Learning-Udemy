# Apriori running with class apyori

# Data Preprocessing
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None) 
# pd. read_csv thought there was a header
# Need to turn the dataset into a list of a list
transactions = []
for i in range(0,dataset.shape[0]):
    transactions.append([str(dataset.values[i,j]) for j in range (0,dataset.shape[1])])
    
# Training Apriori on the dataset
# Import self made apyori.py
from apyori import apriori
#  min_support
# COnsider product that purchase 3 times a day in a 7-day time frame(a week)
# 3*7/7500 =0.0028=~0.003
# Having high min_confidence will give very trivial result not association result
# Rules contain at least 2 (min_length product)
# Rules true 20% (min_confidence)
rules = apriori(transactions, min_support=0.003,
                min_confidence = 0.2 , min_lift =3, min_length =2)

# Visualizing the results
results = list(rules)
myResults = [list(x) for x in results]