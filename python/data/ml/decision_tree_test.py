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
x_train, x_test, y_train, y_test = train_test_split(x_data_select, iris.target, test_size=0.3,stratify=iris.target)


#%%
#과적합 방지 GridSearchCV
x_train , x_test , y_train , y_test = train_test_split(iris.data, iris.target,
                                                       test_size=0.3,random_state=1,stratify=iris.target)

# DecisionTree Classifier 생성
model= DecisionTreeClassifier(random_state=1)

from sklearn.model_selection import GridSearchCV
'''
과적합을 제어하는데 사용
max_depth : 트리의 최대 깊이
- default = None
트리 깊이를 제한하면 과대적합 감소

min_samples_split : 노드를 분할하기 위한 최소한의 샘플 데이터수 
- Default = 2 
작게 설정할 수록 분할 노드가 많아져 과적합 가능성 증가

min_samples_Lieaf:리프가 되기위한 최소샘플수 dafault=1

max_Leaf_nodes : 최대 리프수 제한

max_features : 참여시킬 독립변수 제한
'''




prams = {'criterion':['gini','entropy'], 
              'max_depth':[None,2,3,4,5,6],               
              'min_samples_split':[2,3,4,5,6], 
              'min_samples_leaf':[1,2,3], 
              'max_features':[None,'sqrt','log2']}

grid_dtree = GridSearchCV(model
             ,param_grid=prams
             ,scoring='accuracy'
             ,cv=5,verbose=1)
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


#%% plot_tree
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
#help(plot_tree)
plt.figure(figsize=(20,15))
plot_tree(estimator,filled=True)
plt.savefig('tree2.png')

#%%

import pandas as pd

from sklearn.model_selection import train_test_split
##########데이터 로드
df = pd.read_excel('https://github.com/cranberryai/todak_todak_python/blob/master/machine_learning/multiple_classification/red_wine_MupYMkf.xlsx?raw=true')

df.info()

x_data_df = df.drop(['quality'], axis=1)
y_data_df = df['quality']

#빈도분석
import seaborn as sns
sns.countplot(y_data_df)

#상관성분석
#품질이 좋을수록 알콜도수가 높은 편을 보인다.
#품질이 좋지 않을수록 휘발산이 높은 편을 보인다.
#고정산과 산성도(pH)는 음의 상관관계를 보인다.
sns.pairplot(vars=["alcohol","fixed acidity" ,
                   "volatile acidity", 
                   "pH","quality"], 
             data=df)
#분산분석
#피처별 평균과 표준편차가 다른 편이다.
df.describe()

#%% 전처리
#중복행 제거
#중복행 개수 240
df.duplicated().sum()
#첫행은 남기고 나머지 삭제
#1599 - 240 = 1359 
#삭제시킨 행 인덱스때문에
#군데군데 인덱스 없다
df.drop_duplicates(inplace=True)
#연속적인 행 인덱스로 리셋
df.reset_index(drop=True,
               inplace=True)


x_data_df = df.drop(['quality'], axis=1)
y_data_df = df['quality']

#%%
#이상치 제거
#1Q
quartile_1 = x_data_df.quantile(0.25)
#3Q
quartile_3 = x_data_df.quantile(0.75)
#IQR =3Q-1Q
IQR = quartile_3 - quartile_1
#1Q- IQR*1.5(ㅗ)보다 작거나 
#3Q+ IQR*1.5(T)보다 큰 데이터는 이상치(outlier)

condition = (x_data_df < (quartile_1 - 1.5 * IQR)) | (x_data_df > (quartile_3 + 1.5 * IQR))
condition.head(50)
#데이터프레임.any(axis=1): 가로방향으로 하나라도 True이면 True를 출력 
# (예로  False  True -> True)
condition = condition.any(axis=1) #이상한 행 조건
#불리언 값 not
#불리언 시리즈,배열 ~ (not 안됨)
#import numpy as np
#~np.array([False,False])
'''
18    False
19     True

예로 True인 행은 선택 
'''
condition = ~ condition#부정,이상하지 않은 행 조건

#불리언 인덱스 시리즈로 정상행 선택 추출
x_data_df = x_data_df[condition]
y_data_df = y_data_df[condition]

# 정상행이 제거되면서 군데군데 해당 행 인덱스번호가 없다  
# 행 인덱스번호 연속적으로 reset
x_data_df= x_data_df.reset_index(drop=True)
y_data_df = y_data_df.reset_index(drop=True)

#행 개수 1019로 감소
x_data_df.shape[0]


#%% x,y 분리후 오버샘플링
#conda install -c conda-forge imbalanced-learn 
#각 클래스의 개수가 큰 차이가 난 상태로 모델을 학습하면 다수의 클래스 범주로 분류를 많이하게 되는 문제
#불균형(비대칭) 클래스 데이터(imbalanced data) -> 클래스 데이터들간의 균등화
#오버샘플링: 소수 클래스 데이터에 임의의 데이터(예로 랜덤값)를 추가하여 다수 클래스 데이터에 맟춤(붓꽃데이터셋(50개씩))

from imblearn.over_sampling import RandomOverSampler 
sm = RandomOverSampler(random_state=0)
x_data_df , y_data_df = sm.fit_resample(x_data_df,
                                        y_data_df )
#클래스별 데이터 개수 균등분포
sns.countplot(y_data_df)
#%%
from sklearn.preprocessing import StandardScaler
stand= StandardScaler()
x_data = stand.fit_transform(x_data_df)
#%%
x_train, x_test, y_train, y_test = \
train_test_split(x_data, y_data_df, 
                 test_size=0.3, 
                 random_state=0, 
                 stratify=y_data_df)

#%%  
#모델 학습
from sklearn.tree import DecisionTreeClassifier
#클래스 불균형 해결 class_weight
#샘플 수가 상대적으로 적은 클래스에 가중치를 부여(상대적으로 적은 클래스가 더 예측이 잘되는 것으로 가정)
#클래스 별 가중치 값을 사전형식으로 지정 (약간 성능 개선 가능성)
#model = DecisionTreeClassifier(random_state=0,class_weight={3:3, 4:2,5:1, 6:1,7:1.5, 8:3})
model = DecisionTreeClassifier(random_state=0)
model.fit(x_train, y_train)

#모델 평가
print(model.score(x_train, y_train))#1.0
print(model.score(x_test, y_test))#0.612

#모델 최적화 GridSearchCV() 최적화 (★행부족으로 이상치제거는 하지말고 중복행 제거 감안해서 train_test_split(test_size=0.25)로 설정)
prams={'criterion':['gini','entropy'],
       'max_depth':[None,2,3,4,5],
       'min_samples_leaf':[1,2,3],
       'min_samples_split':[2,3,4]}


from sklearn.model_selection import GridSearchCV
grid_dtree = GridSearchCV(model
             ,param_grid=prams
             ,scoring='accuracy'
             ,cv=5)

grid_dtree.fit(x_train,y_train)

#GridSearchCV 최적 파라미터
print(grid_dtree.best_params_)
#GridSearchCV 최고 정확도
print(grid_dtree.best_score_)#0.870
'''
class_weight후 0.571
'''

#최적의 예측 모델
estimator=grid_dtree.best_estimator_
y_predict=estimator.predict(x_test)

#테스트 데이터 세트 정확도
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predict))
'''
1. class_weight후 0.595
2. 오버샘플링 전후
테스트 점수 accuracy 0.563 -> 0.858
'''
#%% 피처중요도
for k,v in zip(x_data_df.columns,
               estimator.feature_importances_):
    print(k,v)
#막대로 시각화
#알콜 특성이 가장 중요 
#가장 유용한 특성들만 선택하여 학습 특성의 수를 
#줄여서 재학습으로 과적합을 해소할 수 있다.    
import seaborn as sns
#sns.barplot(x=x_data_df.columns,
#            y=estimator.feature_importances_)
#가로막대
sns.barplot(x=estimator.feature_importances_,
            y=x_data_df.columns) 

#DataFrame 기반 내림정렬
d={"feature_names" : x_data_df.columns , 
   "feature_importances" : estimator.feature_importances_}
df = pd.DataFrame(d)
df = df.sort_values(by="feature_importances",
               ascending=False)

sns.barplot(x="feature_importances",
            y="feature_names",
            data = df) 

#Series 기반 내림정렬
feature_importance_values_s = pd.Series(estimator.feature_importances_,
index = x_data_df.columns)
#중요도 값 상위 5개
feature_importance_values_s.sort_values(ascending = False)[:5] 


#%%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

#- quiz
#DecisionTreeClassifier로 최적의 자동차 등급 예측 모델을 찾는다. 
#GridSearchCV 기반의 교차검증으로 최적의 하이퍼 파라미터('criterion', 'max_depth', 'min_samples_split')를 찾아야 한다.
#make_column_transformer()을 적용하고 특성 중요도를 출력한다.

df = pd.read_csv('https://raw.githubusercontent.com/khandelwalpranav05/Car-Evaluation/master/car.data')

x_data = df.drop(['class values'], axis=1)
y_data = df['class values']

#make_column_transformer()을 적용
transformer = make_column_transformer(
    (OneHotEncoder(), ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']),
    remainder='passthrough')

transformer.fit(x_data)
x_data=transformer.transform(x_data)

le = LabelEncoder()
le.fit(y_data)
y_data=le.transform(y_data)


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=10, stratify=y_data)
model= DecisionTreeClassifier(random_state=1)
param_grid = {'criterion':['gini','entropy'], 
         'max_depth':[1,2,3,4,5,6],               
         'min_samples_split':[2,3,4,5,6]}

grid_dtree = GridSearchCV(model,param_grid=param_grid,scoring='accuracy',cv=3)
grid_dtree.fit(x_train,y_train) 
grid_dtree.best_params_ #gini 'max_depth': 6, 'min_samples_split': 2

best_estimator = grid_dtree.best_estimator_

#특성중요도 
print(best_estimator.feature_importances_)