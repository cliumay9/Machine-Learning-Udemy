#Data Preprocessing with R

# Importing the dataset
# Note: Make sure you set the correct working directory
# Use ifelse to fix missing data on $Age and $Salary

dataset = read.csv('Data.csv')
dataset$Age = ifelse(is.na(dataset$Age), ave(dataset$Age, FUN = function(x) mean(x,na.rm=TRUE)), dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary), ave(dataset$Salary, FUN = function(x) mean(x,na.rm=TRUE)), dataset$Salary)

#