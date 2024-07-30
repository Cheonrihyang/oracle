# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:50:40 2024

@author: ORC
"""

#%% 이진분류


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

##########데이터 분석

##########데이터 전처리

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

##########모델 생성

model = LogisticRegression(penalty='none')

##########모델 학습

model.fit(x_train, y_train)

##########모델 검증
model.score(x_train,y_train)
model.score(x_test,y_test)
##########모델 예측

x_test = np.array([
    [4, 6]
])

y_predict = model.predict(x_test)
print(y_predict) #[1]
print(y_predict[0]) #1
label = labels[y_predict[0]]
y_predict = model.predict_proba(x_test)
confidence = y_predict[0][y_predict[0].argmax()]

print(label, confidence) #

x_test = [
    [4, 6], 
    [10, 11]
]

y_predict = model.predict(x_test)
print(y_predict) #[1 1]
print(y_predict[0]) #1
label = labels[y_predict[0]]
y_predict = model.predict_proba(x_test)
confidence = y_predict[0][y_predict[0].argmax()]
print(label, confidence) #
print(y_predict[1]) #1
label = labels[y_predict[1]]
y_predict = model.predict_proba(x_test)
confidence = y_predict[1][y_predict[1].argmax()]
print(label, confidence) #

#%%

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import seaborn as sns


dataset = datasets.load_breast_cancer()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

#데이터프레임 정보 확인
print(df.head())
print(df.shape)
df.info()
desc=df.describe()

#0과 1의 갯수가 비슷함 타겟정규화는 필요 없을듯
df['target'].value_counts()

x_data = dataset.data
y_data = dataset.target


#x데이터 정규화
scaler = StandardScaler()
x_data_scaled = scaler.fit_transform(x_data)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data , test_size=0.3, random_state=0)

model = LogisticRegression()
model.fit(x_train,y_train)

model.score(x_train,y_train)
model.score(x_test,y_test)

#'lbfgs'이 L2 지원
params={'C':[0.1,1,10],
        'penalty':['l1','l2'],
        'solver':['saga','liblinear'],
        'max_iter':[100,500]        
        }


from sklearn.model_selection import GridSearchCV
grid_clf = GridSearchCV(model, param_grid=params, cv=5, scoring='accuracy',verbose=1)
grid_clf.fit(x_train,y_train)

grid_clf.best_params_
grid_clf.best_score_
grid_clf.score(x_test,y_test)



#%% 다항(다중)로지스틱 회귀(다중클래스 회귀분류)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

#'A', 'B', 'C' 범주(카테고리,그룹,클래스)값이
# 각각 0,1,2 
y_data = [2, 2, 2, 1, 1, 2, 0, 0, 0, 1, 0, 2]

labels = ['A', 'B', 'C']

##########데이터 분석

##########데이터 전처리

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777, stratify=y_data)

##########모델 생성

model = LogisticRegression(multi_class='multinomial')

##########모델 학습

model.fit(x_train, y_train)

##########모델 검증

##########모델 예측
y_predict = model.predict(x_test)

x_test = [
    [4, 6]
]

y_predict = model.predict(x_test)
print(labels[y_predict[0]]) #B




#%% pipeline

#전처리 각 단계,모델링,학습을 한번에 연결처리

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np

##########데이터 로드

df = pd.read_csv('https://raw.githubusercontent.com/cranberryai/todak_todak_python/master/machine_learning/multiple_classification/car.data')
df.info()
##########데이터 분석

##########데이터 전처리
#['buying' 구매가격, 
#'maint' 유지보수비용, 
#'doors' 문개수
#, 'persons' 좌석수, 
#'lug_boot',짐칸 크기 
#'safety'] 안전

#class values 평가 등급

x_data = df.drop(['class values'], axis=1)
y_data = df['class values']

print(x_data.head())
'''
    buying  maint  doors persons lug_boot safety
744   high    med  5more       4      big    low
88   vhigh  vhigh  5more       2      big    med
814   high    low      4       2      med    med
805   high    low      3    more      med    med
968    med  vhigh  5more    more      med   high
'''
print(x_data.columns) #

transformer = make_column_transformer(
    (OneHotEncoder(), ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']),
    remainder='passthrough')


le = LabelEncoder()
le.fit(y_data)
print(le.classes_) #['acc' 'good' 'unacc' 'vgood']
#labels = le.classes_
labels = ['수용가능', '좋음', '수용불가', '매우좋음']
y_data = le.transform(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777, stratify=y_data)

##########모델 생성

model = make_pipeline(transformer, LogisticRegression())
#transformer만 존재할시 fit_transform사용
#예측모델이랑 혼용시 fit사용

##########모델 학습
model.fit(x_train, y_train)

##########모델 검증

print(model.score(x_train, y_train)) #

print(model.score(x_test, y_test)) #0.8805394990366089

##########모델 예측

x_test = [
    ['vhigh', 'vhigh', '2', '2', 'small', 'low']
]
x_test = pd.DataFrame(x_test, columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

y_predict = model.predict(x_test)


label = labels[y_predict[0]]
y_predict = model.predict_proba(x_test)
confidence = y_predict[0][y_predict[0].argmax()]

print(label, confidence) #


#%%

from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats,polyval
import numpy as np

avg=[0.332,0.335,0.298,0.305,0.313,0.336,0.316,0.268,0.287,0.297,
     0.289,0.285,0.294,0.301,0.339,0.285,0.301,0.266,0.283,0.323,
     0.302,0.319,0.297,0.312,0.288,0.281,0.283,0.295,0.249,0.249,
     0.320,0.290,0.271,0.253,0.263,0.268,0.233,0.293,0.292,0.265]
ops=[0.856,0.842,0.929,0.870,0.893,0.901,0.786,0.767,0.746,0.936,
     0.812,0.707,0.764,0.734,0.836,0.807,0.825,0.713,0.800,0.846,
     0.887,0.877,0.741,0.852,0.773,0.787,0.815,0.779,0.708,0.777,
     0.739,0.811,0.692,0.819,0.779,0.703,0.668,0.747,0.774,0.703]
df=pd.DataFrame()
df['avg']=avg
df['ops']=ops


Q1 = df['avg'].quantile(0.25)
Q3 = df['avg'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data_clean = df[(df['avg'] >= lower_bound) & (df['avg'] <= upper_bound)]
x_train, x_test, y_train, y_test = train_test_split(data_clean['avg'], data_clean['ops'], test_size=0.3, random_state=1)
x_train = x_train.to_numpy().reshape(-1, 1)
x_test = x_test.to_numpy().reshape(-1, 1)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

model = LinearRegression()
model.fit(x_train,y_train)

print(f'학습 점수 : {model.score(x_train,y_train)*100}점')
print(f'테스트데이터 점수 : {model.score(x_test,y_test)*100}점')


print(f'3할타자 예측 ops: {model.predict([[0.3]])[0]}')
plt.title('2023 kbo avg/ops')
plt.xlabel('avg')
plt.ylabel('ops')
sns.regplot(x=df['avg'], y=df['ops'])


#%%

war=[7.95,7.12,6.86,6.31,5.91,5.79,5.74,4.63,4.48,4.48,4.20,3.54,3.41,
     3.40,3.20,2.90,1.64]

era=[2.0,2.65,2.39,2.54,2.78,2.67,3.28,3.45,3.53,3.24,3.60,3.58,3.83,
     3.42,4.30,3.54,5.23]

df=pd.DataFrame()
df['war']=war
df['era']=era


Q1 = df['era'].quantile(0.25)
Q3 = df['era'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_clean = df[(df['era'] >= lower_bound) & (df['era'] <= upper_bound)]


x_train, x_test, y_train, y_test = train_test_split(data_clean['era'], data_clean['war'], test_size=0.3, random_state=777)
x_train = x_train.to_numpy().reshape(-1, 1)
x_test = x_test.to_numpy().reshape(-1, 1)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

model = make_pipeline(StandardScaler(), LinearRegression())

model.fit(x_train,y_train)

model.score(x_train,y_train)
model.score(x_test,y_test)
model.predict(x_train)


plt.title('2023 kbo era/war')
plt.xlabel('era')
plt.ylabel('war')
sns.regplot(x=df['era'], y=df['war'])
sns.regplot(x=x_train, y=model.predict(x_train))


#%% RandomizedSearchCV


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


df = pd.read_csv('https://raw.githubusercontent.com/cranberryai/todak_todak_python/master/machine_learning/multiple_classification/car.data')

x_data = df.drop(['class values'], axis=1)
y_data = df['class values']

transformer = make_column_transformer(
    (OneHotEncoder(), ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']),
    remainder='passthrough')

transformer.fit(x_data)
x_data=transformer.transform(x_data)

le = LabelEncoder()
le.fit(y_data)
y_data=le.transform(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=10, stratify=y_data)

#기본 LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)

model.score(x_train,y_train) #0.91
model.score(x_test,y_test) #0.92


#GridSearchCV 적용
param_grid = {
    'C':[0.1,0.5,1,5,10],
    'max_iter':[0,10,50,100,500,1000]      
}

#GridSearchCV는 적은 데이터에서 최적을 찾을때 RandomizedSearchCV는 데이터가 너무 많을때
#grid_search = GridSearchCV(model, param_grid=param_grid,scoring='accuracy',verbose=1) 
grid_search = RandomizedSearchCV(model, param_distributions=param_grid,
                                 scoring='accuracy',verbose=1,random_state=1,
                                 n_iter=3)
#random_state=시드값 n_iter횟수제한

grid_search.fit(x_train,y_train)

#결과출력
print(grid_search.cv_results_)

grid_search.score(x_train,y_train) #0.94
grid_search.score(x_test,y_test) #0.94

estimator=grid_search.best_estimator_
y_predict=estimator.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predict))

#%%

iris = load_iris()
X, y = iris.data, iris.target
