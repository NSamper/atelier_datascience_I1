# Packages génériques
import sys
import os
import importlib
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


dataset = "insurance.csv"

df = pd.read_csv(dataset)

print(df.head)

model = LinearRegression()

y = df['charges']
x = df[["bmi"]]

model.fit(x, y)

theta0 = model.intercept_
theta1 = model.coef_[0]

print("Coefficient theta0 (intercept) :", theta0)
print("Coefficient theta1 (pente) :", theta1)

# Tracer la droite de régression
plt.plot(df['bmi'], df['charges'], 'ro', markersize=4)
plt.xlabel('bmi')
plt.ylabel('charges')
plt.plot([15, 55], [theta0, theta0 * theta1], linestyle='--', c='#000000')

plt.show()