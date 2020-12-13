# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:47:37 2020

@author: Zeno
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd 


data = pd.read_csv("D:/MACHINE LEARNING EXAMPLES/Multiple Regression/multiple_regression_dataset.csv",sep=";")

y= data.maas.values.reshape(-1,1)
x= data.iloc[:,[0,2]].values

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0 : ",multiple_linear_regression.intercept_)
print("b1,b2:",multiple_linear_regression.coef_)

multiple_linear_regression.predict(np.array([[10,35],[5,35]]))