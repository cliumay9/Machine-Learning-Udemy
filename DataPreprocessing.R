#Data Preprocessing with R

# Importing the dataset
# Note: Make sure you set the correct working directory
# Use ifelse to fix missing data
dataset = read.csv('Data.csv')
dataset$Age = ifelse(is.na(dataset$Age), ave(dataset$Age, FUN = function(x) mean(x,na.rm=TRUE)), dataset$Age)

