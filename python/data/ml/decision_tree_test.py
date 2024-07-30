# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:32:04 2021

@author: selene
"""

#동전 앞면과  주사위수 1
#확률 1/2과 1/6의 정보량(엔트로피)
#주사위수1이 정보량(엔트로피) 더크다
import numpy as np
-np.log2(1/2) #1

-np.log2(1/6) #2.58
#%%
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777)

##########모델 학습

model = DecisionTreeClassifier()

model.fit(x_train, y_train)

##########모델 검증

print(model.score(x_test, y_test)) #1.0

##########모델 예측

x_test = [
    [7, 5]
]

y_predict = model.predict(x_test)
print(labels[y_predict[0]]) #pass

#%%

import numpy as np
import matplotlib.pyplot as plt

# 데이터 및 모델
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

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777)

model = DecisionTreeClassifier(random_state=777)
model.fit(x_train, y_train)

# 결정 경계 시각화 함수
def plot_decision_boundary(clf, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    plt.contourf(x1, x2, y_pred, alpha=0.3)
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs", label="Class 0")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^", label="Class 1")
    plt.axis(axes)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

# 데이터 분포와 결정 경계 시각화
plt.figure(figsize=(8, 6))
plot_decision_boundary(model, x_data, y_data, axes=[0, 10, 0, 14])
plt.title("Decision Tree Decision Boundary")
plt.show()

#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 붓꽃의 품종 분류하기 모델
# 붓꽃 데이터를 로딩하고, 학습과 테스트 데이터 셋으로 분리

#iris맵 
iris= load_iris()

print("iris.keys : \n{}".format(iris.keys()))
#dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])

print("특성 이름 : {}\n".format(iris.feature_names))
#특성 이름 : ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width 
#(cm)']

# iris.target_names #array(['setosa', 'versicolor', 'virginica'], dtype='<U10') 이상함


#피처일부로 데스트
#x = iris.data[:, 2:] #꽃잎의 길이,폭 
#y = iris.target	

iris.data.shape
#x_train , x_test , y_train , y_test = train_test_split(x, y,    test_size=0.25,  random_state=11)

x_train , x_test , y_train , y_test = train_test_split(iris.data, iris.target,
                                                       test_size=0.3,random_state=1)

# DecisionTree Classifier 생성
#model= DecisionTreeClassifier()
# 지니계수 대신에 entropy
model= DecisionTreeClassifier(criterion='gini', random_state=1)

# DecisionTreeClassifer 학습 
model.fit(x_train , y_train)
# DecisionTreeClassifer 예측
y_predict = model.predict(x_test)

from sklearn.metrics import *
print("훈련 세트 정확도: {:.3f}".format(model.score(x_train, y_train)))# 1.000
#print("테스트 세트 정확도: {:.3f}".format(accuracy_score(x_test,y_predict)))
print("테스트 세트 정확도: {:.3f}".format(model.score(x_test, y_test)))#0.956




#%%
#특성(피처) 중요도 
print(model.feature_importances_)
#entropy 기준 petal length 특성이 가장 중요하다 
for k,v in zip(iris.feature_names,model.feature_importances_):
    print(k,v)
    
import seaborn as sns
sns.barplot(x=iris.feature_names,y=model.feature_importances_)    


#%%
#모델 기반 특성 선택 (Model based feature selection)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
#feature importance가 지정한 임계치보다 큰 모든 특성 선택
select = SelectFromModel(estimator=DecisionTreeClassifier(), threshold="median") #median, mean, 0.25
#select = SelectFromModel(estimator=RandomForestClassifier(n_estimators=10), threshold="median")
x_data_select = select.fit_transform(iris.data, iris.target)
mask = select.get_support()
np.array(iris.feature_names)[mask]
x_train, x_test, y_train, y_test = train_test_split(x_data_select, iris.target, test_size=0.3)


#%%
#과적합 방지 GridSearchCV
x_train , x_test , y_train , y_test = train_test_split(iris.data, iris.target,
                                                       test_size=0.3,random_state=1)

# DecisionTree Classifier 생성
model= DecisionTreeClassifier(criterion='gini', random_state=1)

from sklearn.model_selection import GridSearchCV
'''
과적합을 제어하는데 사용
max_depth : 트리의 최대 깊이
- default = None
트리 깊이를 제한하면 과대적합 감소

min_samples_split : 노드를 분할하기 위한 최소한의 샘플 데이터수 
- Default = 2 
작게 설정할 수록 분할 노드가 많아져 과적합 가능성 증가

'''
prams = {'criterion':['gini','entropy'], 
              'max_depth':[2,3,4,5,6],               
              'min_samples_split':[2,3,4,5,6], 
              'min_samples_leaf':[1,2,3], 
              'max_features':['sqrt','log2']}

grid_dtree = GridSearchCV(model
             ,param_grid=prams
             ,scoring='accuracy'
             ,cv=3)
grid_dtree.fit(x_train,y_train)
#%%
import pandas as pd
#cv결과표
cv_score = pd.DataFrame(
    grid_dtree.cv_results_)

cv_score[['params','mean_test_score']]
#{'max_depth': 3, 'min_samples_split': 2}
grid_dtree.best_params_
grid_dtree.best_score_
estimator = grid_dtree.best_estimator_
#DecisionTreeClassifier() 경우
#DecisionTreeClassifier(max_depth=3)
#DecisionTreeClassifier(criterion='entropy')경우
#DecisionTreeClassifier(criterion='entropy', max_depth=3)


print("훈련 세트 정확도: {:.3f}".format(estimator.score(x_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(estimator.score(x_test, y_test)))


#%%

#와인분석
#11개의 화학적 측정 결과 데이터를 기반으로 와인의 품질을 0~10사이의 등급으로 분류하는 예측
import pandas as pd

from sklearn.model_selection import train_test_split

##########데이터 로드

df = pd.read_excel('https://github.com/cranberryai/todak_todak_python/blob/master/machine_learning/multiple_classification/red_wine_MupYMkf.xlsx?raw=true')
x_data_df = df.drop(['quality'], axis=1)
y_data_df = df['quality'] #y = quality

print(x_data_df.head())

'''

      fixed acidity  volatile acidity  citric acid  ...    pH  sulphates  alcohol

415             8.6             0.725         0.24  ...  3.32       1.07      9.3

332             8.0             0.580         0.28  ...  3.22       0.54      9.4

1509            7.9             0.180         0.40  ...  3.28       0.70     11.1

431             7.8             0.550         0.35  ...  3.25       0.56      9.2

1320            9.7             0.660         0.34  ...  3.26       0.66     10.1




[5 rows x 11 columns]

'''

print(x_data_df.head().T)

'''

                          415       332      1509     431       1320

fixed acidity           8.6000    8.0000   7.9000   7.8000   9.70000

volatile acidity        0.7250    0.5800   0.1800   0.5500   0.66000

citric acid             0.2400    0.2800   0.4000   0.3500   0.34000

residual sugar          6.6000    3.2000   1.8000   2.2000   2.60000

chlorides               0.1170    0.0660   0.0620   0.0740   0.09400

free sulfur dioxide    31.0000   21.0000   7.0000  21.0000  12.00000

total sulfur dioxide  134.0000  114.0000  20.0000  66.0000  88.00000

density                 1.0014    0.9973   0.9941   0.9974   0.99796

pH                      3.3200    3.2200   3.2800   3.2500   3.26000

sulphates               1.0700    0.5400   0.7000   0.5600   0.66000

alcohol                 9.3000    9.4000  11.1000   9.2000  10.10000

'''

print(x_data_df.columns)

'''

Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

       'pH', 'sulphates', 'alcohol'],

      dtype='object')

'''


import seaborn as sns

#특징값의 분포와 상관관계를 히스토그램과 스캐터플롯으로 나타내면 다음과 같다.
#sns.pairplot(hue="quality", data=df)
sns.pairplot(vars=["alcohol","fixed acidity" , "volatile acidity", "pH","quality"], data=df)

# 표준화
# x_data = x_data_df.to_numpy()
from sklearn.preprocessing import StandardScaler
stand= StandardScaler()
x_data = stand.fit_transform(x_data_df)  # 훈련용 데이터를 표준화한다


##########모델 학습

from sklearn.tree import DecisionTreeClassifier
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data_df, test_size=0.3, random_state=1, stratify=y_data_df)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

##########모델 검증평가

print(model.score(x_train, y_train)) # 1.0
sns.countplot(y_data_df)
#붓꽃 데이터셋에 비해서는 저급,고급쪽이 샘플수가 상대적으로 매우 부족
#->개수분포들이 균형이 되도록 오버샘플링(샘플추가) 
print(model.score(x_test, y_test)) #0.61875
#alcohol : 0.180
for name, value in zip(df.columns , model.feature_importances_):

    print('{0} : {1:.3f}'.format(name, value))
