# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:09:09 2020
@author: Zeno
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
"""
In this example I wanted to show prediction of y by 2 different x values
This is a game which x1 is level of the character and x2 is damage of the character
y is how many coins we can collect by depends on these variables
as you see, the coin is increased along with x1 and x2
I want to predict that how many coins we can collect when x1 is 50 and x2 is 35
"""
x = [[0,1],[5,7],[15,12],[25,19],[35,28,],[45,33],[55,40],[60,48]] #character level & character damage
y = [4,5,20,14,32,22,38,43] #coins
""" we made them numpay array"""
x,y = np.array(x),np.array(y) 
"""fitting x and y into our multiple linear regression model"""
multiple_linear_model = LinearRegression()
multiple_linear_model.fit(x,y)

""" let's look at our constant and coefficient"""
print("intercept : ",multiple_linear_model.intercept_,"\n")
print("coeeficient : ",multiple_linear_model.coef_,"\n")
""" and now we want to predict how many coins we can collect when x1 =50 and z2=35"""
print("prediction : ",multiple_linear_model.predict([[50,35]]))
