# Poly Regression

## Beginning of Data Preprocessing ##
dataset = read.csv('Position_Salaries.csv')
# Reset the dataset because we don't need the first column
dataset = dataset[2:3] # Ignore first column
## END of Data Preprocessing ##

## Fitting Linear regression to the dataset
lin_reg = lm(Salary ~ ., data = dataset)

## FItting Poly Regression to the dataset
# Add column into the dataset
dataset$Level2 = (dataset$Level)^2
dataset$Level3 = (dataset$Level)^3
dataset$Level4 = (dataset$Level)^4
poly_reg = lm(Salary ~., data = dataset)

# Visualizing the Linear Regression results
# install.packages('ggplot2) if you dont have ggplot2 installed
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') + 
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') + 
  ggtitle('Salary vs Level (Linear Regression)')+
  xlab('Level')+
  ylab('Salary')

# Visualizing the Poly Regression results
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') + 
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') + 
  ggtitle('Salary vs Level (Poly Regression)')+
  xlab('Level')+
  ylab('Salary')

# Predict new result with Linear Regression, lin_reg
y_pred = predict(lin_reg, data.frame(Level = 6.5))

# Predict new result with Poly Reg, poly_reg
y_poly_red = predict(poly_reg, data.frame(Level = 6.5,
                                          Level2= 6.5^2,
                                          Level3= 6.5^3,
                                          Level4=6.5^4))
