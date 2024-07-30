# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:46:58 2024

@author: ORC
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 예시 데이터
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])

# 다항 피처 변환
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

# 다항 회귀 모델 훈련
model = LinearRegression()
model.fit(x_poly, y)

# 예측
x_range = np.linspace(1, 5, 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_range_pred = model.predict(x_range_poly)

# 시각화
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x_range, y_range_pred, color='red', label='Polynomial Regression')
plt.xlabel('Study Hours')
plt.ylabel('Test Score')
plt.title('Polynomial Regression Example')
plt.legend()
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 예시 데이터
x = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([3, 5, 7, 9, 11])

# 다항 피처 변환
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

# 다항 회귀 모델 훈련
model = LinearRegression()
model.fit(x_poly, y)

# 예측
x_range = np.linspace(1, 5, 100).reshape(-1, 2)
x_range_poly = poly.transform(x_range)
y_range_pred = model.predict(x_range_poly)

# 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 원래 데이터
ax.scatter(x[:, 0], x[:, 1], y, color='blue', label='Actual Data')

# 회귀 평면
ax.plot_trisurf(x_range[:, 0], x_range[:, 1], y_range_pred, color='red', alpha=0.5, label='Polynomial Regression')

ax.set_xlabel('x1 (Study Hours)')
ax.set_ylabel('x2 (Class Participation)')
ax.set_zlabel('y (Test Score)')
plt.title('Polynomial Regression Example with Two Features')
plt.legend()
plt.show()