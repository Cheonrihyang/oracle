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

