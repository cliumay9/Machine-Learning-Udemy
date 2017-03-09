# Eclat.R
setwd("~/Desktop/Ex_Files_DSF_DataMining/Machine-Learning-Udemy/Eclat")

# Data Preprocessing
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep =',', rm.duplicates= TRUE)
summary(dataset)
itemFrequencyPlot(dataset,topN =10)

# Training apriori on the dataset
arules = apriori(data = dataset, parameter = list(support =0.004, confidence=0.2))

# Visualizing apriori's rules reuslts
inspect(sort(arules, by = 'lift')[1:10])

# Training eclat on the dataset
erules = eclat(data = dataset, parameter = list(support =0.004, minlen = 2))

# Visualizing eclat sets reuslts
inspect(sort(erules, by = 'support')[1:10])