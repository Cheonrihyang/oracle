# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:17:57 2024

@author: ORC
"""
#%%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats,polyval
import seaborn as sns

y=np.random.randn(10).reshape(5,2)
df=pd.DataFrame(y)
df.columns=['x','y']
#공분산
df['x'].cov(df['y'])


#데이터가 정규성을 보이는 경우
df['x'].corr(df['y'],method='pearson')
#피어슨

#데이터가 정규성을 보이지 않는 경우
df['x'].corr(df['y'],method='spearman')
#스피어만

#데이터가 정규성을 보이지 않는 경우
#표본데이터가 적고 동점이 많은 경우
df['x'].corr(df['y'],method='kendall')
#켄달

#상관계수 제곱시 결정계수:설명력

df.corr()
plt.scatter(df['x'], df['y'])


#유의확률
import scipy
scipy.stats.pearsonr(df['x'], df['y'])


#%%

# 식료품 물가상승률 (요소 개수 12) 
x = [3.52, 2.58, 3.31, 4.07, 4.62, 3.98, 4.29, 4.83, 3.71, 4.61, 3.90, 3.20]
# 엥겔지수
y = [2.48, 2.27, 2.47, 2.77, 2.98, 3.05, 3.18, 3.46, 3.03, 3.25, 2.67, 2.53]
df=pd.DataFrame()
df['식료품 물가상승률']=x
df['앵겔지수']=y
plt.scatter(df['식료품 물가상승률'], df['앵겔지수'])

model = stats.linregress(x,y)
model.slope #계수(약 0.5)
model.intercept #(약 0.91)
#y=0.5x+0.91 #물가상승률이 1증가시 엥겔지수가 0.5증가

model.rvalue**2 #결정계수
#약 0.8(엥겔지수 변동율 약 80%정도 잘 설명한다)

model.pvalue
#0.05미만일때 통계적으로 유의미함.

#%%

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
plt.title('2023년 kbo리그 타자war 상위40인 타율,ops 상관관계')
plt.xlabel('avg')
plt.ylabel('ops')
sns.regplot(x=df['avg'], y=df['ops'])

model = stats.linregress(avg,ops)
model.slope
model.intercept
print(f'결정계수 {model.rvalue**2}')
print(f'유의확률 {model.pvalue}')

predict=np.poly1d([model.slope, model.intercept],variable = 'x')
print(predict)
predict(0.22)

#%% 퀴즈

from scipy import stats,polyval
import matplotlib.pyplot as plt


#출석시간
x=[2,5,6,8,12,5,9,9,8]
#시험점수
y=[3,5,7,10,12,7,13,13,12]

model = stats.linregress(x,y)
model.rvalue**2
model.pvalue
plt.scatter(x, y)
plt.xlabel('출석시간')
plt.ylabel('시험점수')
predict=np.poly1d([model.slope, model.intercept],variable = 'x')
predict(7)
print(predict)

#%%
import pandas as pd
import statsmodels.formula.api as smf

x = [2, 5, 6, 8, 12, 5, 9, 9, 8]

y = [3, 5, 7, 10, 12, 7, 13, 13, 12]
df = pd.DataFrame({"x":x,"y":y})
#선형회귀모델 설정
model=smf.ols(data=df,formula='y~x-1') #formula=종속변수 ~ 독립변수
#-1은 절편을 포함하지 않는다.

#회귀모델 학습
reg_model = model.fit()
reg_model.params
# Intercept    0.967
# x            1.14
# y = 1.14 x + 0.967
#예측 
reg_model.predict(dict(x=3.52))
ry=reg_model.predict(df['x'])

#평가
#R-squared:      0.815
reg_model.summary()
#시각화
sns.regplot(x='x',y='y',data=df)
sns.regplot(x='x',y=ry,data=df)

#%% statsmodels 패키지 기반 다항회귀 분석
#엥겔지수 = 식료품 물가상승률 x + 가계소득 x2
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
x = [3.52, 2.58, 3.31, 4.07, 4.62, 3.98, 4.29, 4.83, 3.71, 4.61, 3.90, 3.20]
#가상 가계소득 
x2 = [4.52, 2.58, 3.31, 4.05, 4.62, 3.58, 4.29, 4.63, 3.71, 4.61, 3.80, 3.20]
y = [2.48, 2.27, 2.47, 2.77, 2.98, 3.05, 3.18, 3.46, 3.03, 3.25, 2.67, 2.53]
df = pd.DataFrame({"x":x,"x2":x2,"y":y})

#선형회귀모델 설정
#smf는 상수항(절편 초기값)이 자동 추가
model=smf.ols(data=df,formula='y~ x + x2') #formula=종속변수 ~ 독립변수
#sm는 상수항(절편 초기값)이 수동 추가
#cx = sm.add_constant(df[['x','x2']]) #x에 원소가 1인 상수항(절편 초기값)이 추가
#model=sm.OLS(y,cx) #종속변수 , 절편 초기값이 추가된 독립변수
reg_model = model.fit()
#예측
reg_model.params
# const    1.033180
# x        0.693390
# x2      -0.225672 #가계 소득이 높아질수록 엥겔지수의 비중이 감소

#smf 예측방식 
reg_model.predict(dict(x=3.52,x2=4.52)) #2.453877
#sm 예측방식
#reg_model.predict([[1,3.52,4.52]]) #2.453877

#평가
reg_model.summary()

reg_model.resid.plot()
from matplotlib import pyplot as plt
import seaborn as sns
plt.subplot(1,2,1)#1행 2열의 1번째
sns.regplot(x=df['x'],y=df['y'],color='red')
plt.subplot(1,2,2)
sns.regplot(x=df['x2'],y=df['y'],color='green')

#%%로지스틱 회귀분석

import pandas as pd
import statsmodels.api as sm

score = [56, 50, 30, 62, 69, 55, 72, 44, 51, 64, 60, 59, 68, 72, 80, 93, 65, 94, 81, 70, 92, 90, 77, 78, 100]
_pass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

data = pd.DataFrame(
    {"score": score,"_pass": _pass}
)

##로지스틱회귀모델
#결과값이 0과 1로 이루어졌다면 선형으로 하면 어색하다

model= sm.Logit.from_formula(data=data,formula='_pass ~ score')
logis_model = model.fit()
logis_model.summary()

logis_model.predict(dict(score=75))
logis_model.params

#시각화
sns.scatterplot(x=score, y=_pass, label="Actual Data")
plt.grid(True)
plt.scatter(score, logis_model.predict(data['score']),marker='x')
plt.show()

#%%#다중 로지스틱 회귀분석

import pandas as pd
import seaborn as sns
score1 = [56, 50, 30, 62, 69, 55, 72, 44, 51, 64, 60, 59, 68, 72, 80, 93, 65, 94, 81, 70, 92, 90, 77, 78, 100]
score2 = [56, 60, 61, 67, 69, 55, 70, 44, 51, 64, 60, 50, 68, 72, 90, 93, 85, 74, 81, 88, 92, 97, 77, 78, 98]
_pass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

data = pd.DataFrame(
    {"score1": score1,"score2": score2,"_pass": _pass}
)

##로지스틱회귀모델
model= sm.Logit.from_formula(data=data,formula='_pass ~ score1 + score2')
logis_model = model.fit()
logis_model.summary()

plt.grid(True)
#예상값을 새로운 열로 만듬
data['pred_pass']=logis_model.predict(data[['score1','score2']])
#boxplot형으로 출력
sns.boxplot(x='_pass',y='pred_pass',data=data)

#%% 퀴즈

#상품매장 방문고객의 간단 프로필기반의 상품구매 여부 예측(로지스틱회귀 분석)
#나이,월수입에 따른 상품구매여부(범주형) 분류
#상품구매여부 = 0:비구매, 1:구매

#모델링, 모델평가, 예측 ,예측값과 실제값을 비교 시각화

import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('data/buy.csv',encoding='utf-8')
'''
df['상품구매여부'].sum()
df['상품구매여부'].count()
df['상품구매여부'].size
df['상품구매여부'].unique()
df['상품구매여부'].value_counts()
sns.countplot(x='상품구매여부',data=df)
'''

plt.rc('font', family='Malgun Gothic') 

model = sm.Logit.from_formula(data=df,formula='상품구매여부 ~ 나이 + 월수입')
logis_model = model.fit()
plt.figure(figsize=(12,5))

x=[df['나이'],df['월수입'],logis_model.predict(df[['나이','월수입']])]
y=[logis_model.predict(df[['나이','월수입']]),logis_model.predict(df[['나이','월수입']]),df['상품구매여부']]
xlabel=['나이','월수입','구매예측지수']

    
plt.subplot(1,3,1)
plt.grid(True)
plt.xlabel('나이')
plt.ylabel('구매여부')
sns.scatterplot(x=df['나이'], y=df['상품구매여부'])
plt.scatter(x=df['나이'], y=logis_model.predict(df[['나이','월수입']]),marker='x')

plt.subplot(1,3,2)
plt.grid(True)
plt.xlabel('월수입')
plt.ylabel('구매여부')
sns.scatterplot(x=df['월수입'], y=df['상품구매여부'])
plt.scatter(x=df['월수입'], y=logis_model.predict(df[['나이','월수입']]),marker='x')

plt.subplot(1,3,3)
plt.grid(True)
plt.xlabel('구매예측지수')
plt.ylabel('구매여부')
plt.scatter(x=logis_model.predict(df[['나이','월수입']]),y=df['상품구매여부'],marker='x')

plt.show()


#%% LinearRegression모델생성

from sklearn.linear_model import LinearRegression

##########데이터 로드

x_data = [2, 3, 3, 5, 7, 2, 8, 9, 6, 9, 6, 2] 
y_data = [3, 5, 7, 10, 12, 7, 13, 13, 12, 13, 12, 6]

##########데이터 분석

#1차배열을 2차배열로 변환
import numpy as np
x_data = np.array(x_data).reshape(-1,1)

##########데이터 전처리

##########모델 생성

model = LinearRegression()

##########모델 학습

model.fit(x_data, y_data) #LinearRegression 모델에 x_data, y_data 데이터 적용 (수식 계산)

##########모델 검증

##########모델 예측

x_data = [[4]]

y_predict = model.predict(x_data) #배치 예측
print(y_predict) #[7.98571429]
print(y_predict[0]) #7.985714285714285

x_data = [[4], [10]]

y_predict = model.predict(x_data)
print(y_predict) #[ 7.98571429 15.34489796]
print(y_predict[0]) #7.985714285714285
print(y_predict[1]) #15.344897959183676

#%% 평가

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

x = [3.52, 2.58, 3.31, 4.07, 4.62, 3.98, 4.29, 4.83, 3.71, 4.61, 3.90, 3.20]
#가상 가계소득 
x2 = [4.52, 2.58, 3.31, 4.05, 4.62, 3.58, 4.29, 4.63, 3.71, 4.61, 3.80, 3.20]
y = [2.48, 2.27, 2.47, 2.77, 2.98, 3.05, 3.18, 3.46, 3.03, 3.25, 2.67, 2.53]
df = pd.DataFrame({"x":x,"x2":x2,"y":y})
model = LinearRegression()
model.fit(df[['x','x2']],df['y'])

model.coef_
model.intercept_

model.predict([[3.52,4.52]])
y_preds= model.predict(df[['x','x2']]) 


# 모델 성능평가 mse(대략 예측값과 실제값의 차이로 0에 가까울 수록 좋은 성능) , rmse 
from sklearn.metrics import mean_squared_error,r2_score

#학습평가지표
#mse mean_squared_error(y,y예측값)
mse= mean_squared_error(y ,y_preds)#0.02
np.sqrt(mse)#0.14


#예측평가지표
#r2_score(y,y예측값)
r2_score(y, y_preds) #0.8359 (대략 1- mse )

#모델로 예측평가지표 구하기(모델.score(x,y))
model.score(df[['x','x2']], df['y'])




#summary평가시 x2 p값이 0.17로 매우높고 R-squared와 Adj. R-squared의 차이가 큼
model=smf.ols(data=df,formula='y ~ x + x2')
reg_model = model.fit()
reg_model.summary()

#%%퀴즈

#사이킷런 LinearRegression을 이용한 toluca_company 
#선형회귀분석(시각적 탐색, 모델링, 예측, 다양한 평가하기)  
#Lot_size : 제품크기(x), Work_hours : 작업시간(y)

from sklearn.linear_model import LinearRegression
import numpy as np


df = pd.read_csv('data/toluca_company.csv') ## 데이터 불러오기

df[['Lot_size', 'Work_hours']] = df['LotSize\tWorkHours'].str.split('\t', expand=True).astype(float)
df.drop('LotSize\tWorkHours', axis=1, inplace=True)

Lot_size = np.array(df['Lot_size']).reshape(-1, 1)
Work_hours = df['Work_hours']

model=LinearRegression()
model.fit(Lot_size, Work_hours)

Work_hours2 = model.predict(Lot_size)

print(mean_squared_error(Work_hours, Work_hours2))
print(r2_score(Work_hours, Work_hours2))

plt.scatter(Lot_size, Work_hours)
plt.plot(Lot_size, Work_hours2)
plt.show()

#%%
#보스턴 주택 가격 예측
import numpy as np
import pandas as pd
from sklearn import datasets
#
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = datasets.load_boston()
x_data = dataset.data
y_data = dataset.target
#print(x_data.shape) #(506, 13)
#print(y_data.shape) #(506,)

####################
#train_test_split
#train과 test 데이터 서브셋으로 분할
#test_size=테스트 데이터셋의 비율
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data,
                                                                    test_size=0.3,
                                                                    random_state=1)

#%%
#기본선형회귀모델
estimator = LinearRegression()

estimator.fit(x_train, y_train)

y_predict = estimator.predict(x_train) 
score = metrics.r2_score(y_train, y_predict)
print(score) #0.71

y_predict = estimator.predict(x_test) 
score = metrics.r2_score(y_test, y_predict)
print(score) #0.78

#%%
#DataFrame기반 보스턴 주택 가격 예측
# boston 데이타셋 로드
boston = datasets.load_boston()
boston.keys()
boston.data
boston.feature_names
boston.target
bostonDF = pd.DataFrame(boston.data , columns = boston.feature_names)
bostonDF['PRICE'] = boston.target
bostonDF.info()
bostonDF.describe()
bostonDF.head()
#시각적 탐색분석
#내 특성변수간의 상관관계 플롯
import seaborn as sns
sns.pairplot(data=
   bostonDF[['RM','CRIM','ZN','PRICE']])

lr = LinearRegression()
lr.fit(pd.DataFrame(bostonDF["RM"]) ,
       bostonDF['PRICE'] )
y_preds = lr.predict(pd.DataFrame(bostonDF["RM"]))

# 평가 mse , rmse 
# 모델 성능평가 mse(대략 예측값과 실제값의 차이로 0에 가까울 수록 좋은 성능) , rmse 
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(bostonDF['PRICE'], y_preds)
rmse = np.sqrt(mse)
#R2 = 0.48  PRICE 변동량을 약48% 설명
r2_score(bostonDF['PRICE'], y_preds)
#정확도 점수 = 0.48  RM이 PRICE를 약48% 예측(정답맞춤률)
#y(실측값)가 연속적인 수치인경우(선형회귀) R2를 score로 볼 수 있다

#R2 = score(x,y실값) 
lr.score(pd.DataFrame(bostonDF["RM"]) ,bostonDF['PRICE']  )
#y^(예측값)인경우(선형회귀) R2는 1
lr.score(pd.DataFrame(bostonDF["RM"]) ,y_preds  ) #1.0