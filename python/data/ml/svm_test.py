# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:11:16 2021

@author: selene
"""

#%% svm(서포트벡터머신)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

##########데이터 로드

x_data = [
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
]
y_data = [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0]

labels = ['fail', 'pass']

##########데이터 분석

##########데이터 전처리

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777, stratify=y_data)

##########모델 학습
#선형
#model = SVC(kernel='linear', C=1.0)
model = SVC(kernel='rbf', C=5, gamma=0.5)
#model = SVC(kernel='poly', C=0.8, gamma='auto')
model.fit(x_train, y_train)

##########모델 검증

print(model.score(x_train, y_train)) #1.0
print(model.score(x_test, y_test)) #1.0


#%%
from sklearn.model_selection import train_test_split
from sklearn.svm \
import SVC
from sklearn.datasets import load_iris

iris= load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=777
                                                    , stratify=iris.target)
model= SVC()
model.fit(x_train, y_train) 


print(model.score(x_train , y_train))
print(model.score(x_test  , y_test))

y_predict \
= model.predict(x_test)

from sklearn.metrics \
    import accuracy_score

print(accuracy_score(y_test
               ,y_predict))    

#%%
#이진분류에 적합(두개 품종 분리)
iris= load_iris()
x_data=iris.data[0:100,:]
x_data.shape
#2품종 
y_data=iris.target[0:100]
y_data.shape

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777
                                                    , stratify=y_data)
model= SVC()

model.fit(x_train, y_train) 


print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

y_predict= model.predict(x_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test ,y_predict))    



#%%

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#상품매장 방문고객의 간단프로필 기반 구매여부 분류예측
#DecisionTree, KNN ,SVM으로 판별하는 분류모델을 생성 후 
#각 모델별 훈련정확도와 테스트정확도를 출력하고 비교분석한다.(data/buy2.csv)

#- 고려사항
#1. 나이,월수입,상품구매여부 한글컬럼명의 age,income,buy영문화 후 
#age별 구매여부와 income별 구매여부 개수를 시각화한다.
#2. (23,,비구매)행은 null이 존재하기 때문에 제거
#3. (2,200,구매)행은 나이가 2이다. 이상치로 보고 제거
#4. 피처 크기 정규화


df = pd.read_csv('data/buy2.csv',encoding='utf-8')
df.columns=['age','income','buy']
df.dropna(how='any',inplace=True)

Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[((df['age'] >= lower_bound) & (df['age'] <= upper_bound))]

Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[((df['income'] >= lower_bound) & (df['income'] <= upper_bound))]


x_data = df.drop('buy', axis=1)
y_data = df['buy']

le = LabelEncoder()
le.fit(y_data)
y_data=le.transform(y_data)

sns.boxplot(x='age',hue='buy',y='income',data=df)
plt.scatter(x_data['age'], y_data)
plt.scatter(x_data['income'], y_data)

plt.legend()
plt.show()



stand = StandardScaler()
stand.fit(x_data)
x_data=stand.transform(x_data)

plt.boxplot(x_data)



DTC= DecisionTreeClassifier()
DTC.fit(x_data,y_data)
print(f'DTC스코어 : {DTC.score(x_data, y_data)}')

knn = KNeighborsClassifier()
knn.fit(x_data,y_data)
print(f'knn스코어 : {knn.score(x_data, y_data)}')

SVC = SVC(kernel='linear')
SVC.fit(x_data,y_data)
print(f'SVC스코어 : {SVC.score(x_data, y_data)}')




df.info()
