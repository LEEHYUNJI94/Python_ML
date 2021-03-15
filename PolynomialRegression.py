# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Linear Regression model 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Polynomial Regression model 
from sklearn.preprocessing import PolynomialFeatures
#차원이 높을 수록 정확해지지만, 너무 높으면 overfit / degree = 4제곱
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising 

'''
the Linear Regression results
'''
plt.scatter(X, y, color = 'black')
plt.plot(X, lin_reg.predict(X), color = 'red')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

'''
 the Polynomial Regression results
'''
plt.scatter(X, y, color = 'black')
#predict(X)가 아닌 이유 =  single processing이 아님
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'red')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

'''
the Polynomial Regression results (for higher resolution and smoother curve)
'''
X_grid = np.arange(min(X), max(X), 0.2)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'black')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'red')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting with Linear Regression
print(lin_reg.predict([[6.5]]))

# Predicting with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))