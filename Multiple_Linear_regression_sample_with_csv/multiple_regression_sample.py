# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:47:37 2020

@author: Zeno
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd 

# in this sample, I have salary feature which depends on experience of the employee(x1) and age of the employee(x2)
# and I want to predict what is y when x1-x2 are 10,35 and x1-x2 are 5,35
data = pd.read_csv("D:/Machine Learning Works/Multiple Regression/multiple_regression_dataset.csv",sep=";") # reading csv file
y= data.maas.values.reshape(-1,1) # Salary

# with .iloc method, we took all rows and 0's and 2's colummns which are experience and age
x= data.iloc[:,[0,2]].values # Experience and Age

multiple_linear_regression = LinearRegression().fit(x,y) #fitting the data
print("b0 : ",multiple_linear_regression.intercept_) # looking at the constant 
print("b1,b2:",multiple_linear_regression.coef_,"\n")# looking at the coefficient
#now we predict what is y when x1-x2 are 10,35 and x1-x2 are 5,35
print("x1 is 10 and x2 is 35 =",multiple_linear_regression.predict(np.array([[10,35]])),"and x1 is 5 and x2 is 35 = "
                                                                            ,multiple_linear_regression.predict(np.array([[5,35]])))