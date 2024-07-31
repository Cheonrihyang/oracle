# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:51:24 2024

@author: ORC
"""

#%% 정확도 정밀도 재현율 F1-score ROC

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

##########데이터 로드

x_data = np.array([
    [2, 1],
    [3, 2],
    [3, 4],
    [5, 5],
    [7, 5],
    [2, 5],
    [8, 9],
    [9, 10],
    [6, 12],
    [9, 2],
    [6, 10],
    [2, 4]
])
y_data = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0])

labels = ['fail', 'pass']

##########데이터 전처리

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777, stratify=y_data)

##########모델 학습

model = LogisticRegression()

model.fit(x_train, y_train)

##########모델 검증
y_predict = model.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import classification_report

#1. 정확도,정밀도, 재현율, F1-score
print(accuracy_score(y_test , y_predict))
print(precision_score(y_test , y_predict))
print(recall_score(y_test , y_predict))
print(f1_score(y_test,y_predict))

#2. classification_report()는 정밀도, 재현율, F1-score를 구해 분류보고서를 생성

print(classification_report(y_test, y_predict))
'''
              precision    recall  f1-score   support #support :테스트 데이터 수 4개에서 각 클래스에 속한 샘플수

           0       1.00      1.00      1.00         2
           1       1.00      1.00      1.00         2

    accuracy                           1.00         4
   macro avg(단순산술평균,평균의 평균)       1.00      1.00      1.00         4
weighted avg(각 클래스에 속하는 표본의 갯수로 가중치평균)       1.00      1.00      1.00         4
'''
#3.confusion matrix
print(confusion_matrix(y_test, y_predict))


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
# ROC 곡선의 시각화
# 곡선이 가운데 직선에서 좌상단으로 
# 멀어질수록 성능이 좋다.
def roc_curve_plot(y_test, pred_proba_c1):
    #임계값에 따른 FPR, TPR 값을반환 받음
    fprs, tprs, thresholds  = roc_curve(y_test, pred_proba_c1)#roc_curve() 이진분류만 지원
    # ROC곡선을 그래프로 그림
    plt.plot(fprs, tprs, label='ROC')
    # 가운데 대각선 직선을 그림
    # 대각선: 분류 임계(경계)선 일반적으로 임계값은 50% 설정하고
    # 크면 P 작으면 N    
    plt.plot([0,1], [0,1], 'k--', label='Random')      
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    
#예측분류확률
y_predict_proba = model.predict_proba(x_test)
#y_predict_proba[:, 1] 예측확률의 1번 컬럼
roc_curve_plot(y_test, y_predict_proba[:, 1])

from sklearn.metrics import  roc_auc_score
print('roc_auc_score:',roc_auc_score(y_test,y_predict))

#2. 예측확률기반
print('roc_auc_score:',roc_auc_score(y_test,y_predict_proba[:,1]))
#다중분류인경우
print('roc_auc_score:',roc_auc_score(y_test,y_predict_proba, 
                                     multi_class ='ovr'))
#%% KNeighborsClassifier

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

##########데이터 로드

x_data = np.array([
    [2, 1],
    [3, 2],
    [3, 4],
    [5, 5],
    [7, 5],
    [2, 5],
    [8, 9],
    [9, 10],
    [6, 12],
    [9, 2],
    [6, 10],
    [2, 4]
])
y_data = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0])

labels = ['fail', 'pass']

##########데이터 분석

##########데이터 전처리

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777, stratify=y_data)

##########모델 생성

model = KNeighborsClassifier()

##########모델 학습

model.fit(x_train, y_train)

##########모델 검증

print(model.score(x_train, y_train)) #

print(model.score(x_test, y_test)) #0.75

##########모델 예측

x_test = np.array([
    [4, 6]
])

y_predict = model.predict(x_test)
label = labels[y_predict[0]]
y_predict = model.predict_proba(x_test)
confidence = y_predict[0][y_predict[0].argmax()]

print(label, confidence) #.

# 시각화 위한 그리드 생성
x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 예측 결과를 통해 결정 경계 시각화
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# 훈련 데이터와 테스트 데이터 시각화
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='coolwarm', label='Train Data', edgecolor='k', s=100)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='coolwarm', marker='*', label='Test Data', edgecolor='k', s=200)

# 새로운 데이터 포인트 시각화
new_point = np.array([[4, 6]])
plt.scatter(new_point[:, 0], new_point[:, 1], color='green', label='New Point', marker='X', s=300)

plt.title('KNeighborsClassifier 분류 및 결정 경계')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

#%% 더 쉬운 KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

# 데이터 생성
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=2.0)

# 데이터 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=100)
plt.title("2D 데이터 분포")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# KNeighborsClassifier 모델 생성 및 학습
knn = KNeighborsClassifier(n_neighbors=5)  # k=5로 설정
knn.fit(X, y)

# 새로운 데이터 포인트를 생성하여 예측
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 예측 결과를 시각화
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=100)
plt.title("KNeighborsClassifier 분류 결과 (k=5)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
