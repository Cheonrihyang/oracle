# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:36:41 2021

@author: selene
"""

#%%
#로그 변환기반 타겟 멱함수 분포를 정규분포로 변환
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import Ridge

##########데이터 로드

train_df = pd.read_excel('https://github.com/cranberryai/todak_todak_python/blob/master/machine_learning/regression/carprice_E1SUl6b.xlsx?raw=true', sheet_name='train')
test_df = pd.read_excel('https://github.com/cranberryai/todak_todak_python/blob/master/machine_learning/regression/carprice_E1SUl6b.xlsx?raw=true', sheet_name='test')

##########데이터 분석

##########데이터 전처리

x_train_df = train_df.drop(['가격'], axis=1)
x_test_df = test_df.drop(['가격'], axis=1)
y_train_df = train_df['가격']
y_test_df = test_df['가격']

print(x_train_df.head())
'''
     년식  종류    연비   마력    토크   연료  하이브리드   배기량    중량 변속기
0  2015  대형   6.8  159  23.0  LPG      0  2359  1935  수동
1  2012  소형  13.3  108  13.9  가솔린      0  1396  1035  자동
2  2015  중형  14.4  184  41.0   디젤      0  1995  1792  자동
3  2015  대형  10.9  175  46.0   디젤      0  2497  2210  수동
4  2015  대형   6.4  159  23.0  LPG      0  2359  1935  자동
'''
print(x_train_df.columns) #Index(['년식', '종류', '연비', '마력', '토크', '연료', '하이브리드', '배기량', '중량', '변속기'], dtype='object')

transformer = make_column_transformer(
    (OneHotEncoder(), ['종류', '연료', '변속기']),
    remainder='passthrough')
transformer.fit(x_train_df)
x_train = transformer.transform(x_train_df)
x_test = transformer.transform(x_test_df)
x_test.shape

y_train = y_train_df.to_numpy()
y_test = y_test_df.to_numpy()
y_test.shape
#%%
import seaborn as sns
# 왼쪽으로 치우진 왜곡된 멱함수 분포
sns.distplot(y_train)

import numpy as np
# 왜곡된 y 멱함수분포를 log(y+1)로 정규화 -> 정규분포로 변환
#np.log(0)
y_train= np.log1p(y_train)
y_test= np.log1p(y_test)
sns.distplot(y_train)

##########모델 학습

model = Ridge(alpha=1.0)
model.fit(x_train, y_train)

##########모델 검증
# Ridge + y 분포정규화 이전 :0.88
# Ridge + y 분포정규화 이후 :0.93
print(model.score(x_train, y_train)) #y분포 비정규화 0.883 -> 정규화 0.939
# Ridge + y 분포정규화 이전 :0.77
# Ridge + y 분포정규화 이후 :0.89
# 예측성능 향상, 과적합 감소
print(model.score(x_test, y_test)) #y분포 비정규화 0.77 -> 정규화 0.898

#%%
#MinMaxScaler 피처 스케일링 정규화
train_array = np.arange(0, 11).reshape(-1, 1)
test_array =  np.arange(0, 6).reshape(-1, 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_array)
train_scaled = scaler.transform(train_array)
#scaler.fit(test_array)  넣지말자
test_scaled = scaler.transform(test_array)

#%%
#StandardScaler기반 피처(특성) 정규화
##########데이터 전처리

x_train_df = train_df.drop(['가격'], axis=1)
x_test_df = test_df.drop(['가격'], axis=1)
y_train_df = train_df['가격']
y_test_df = test_df['가격']

print(x_train_df.head())
'''
     년식  종류    연비   마력    토크   연료  하이브리드   배기량    중량 변속기
0  2015  대형   6.8  159  23.0  LPG      0  2359  1935  수동
1  2012  소형  13.3  108  13.9  가솔린      0  1396  1035  자동
2  2015  중형  14.4  184  41.0   디젤      0  1995  1792  자동
3  2015  대형  10.9  175  46.0   디젤      0  2497  2210  수동
4  2015  대형   6.4  159  23.0  LPG      0  2359  1935  자동
'''
print(x_train_df.columns) #Index(['년식', '종류', '연비', '마력', '토크', '연료', '하이브리드', '배기량', '중량', '변속기'], dtype='object')

from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
# StandardScaler객체 생성
scaler = StandardScaler()
#scaler = MinMaxScaler()
# StandardScaler 로 데이터 셋 변환. fit( ) 과 transform( ) 호출.  
transformer = make_column_transformer(
    (scaler,['연비','마력','배기량']),
    (OneHotEncoder(), ['종류', '연료', '변속기']),
    remainder='passthrough')
transformer.fit(x_train_df)
x_train = transformer.transform(x_train_df)
x_test = transformer.transform(x_test_df)


y_train = y_train_df.to_numpy()
y_test = y_test_df.to_numpy()

import seaborn as sns
# 왼쪽으로 치우진 왜곡된 분포
sns.distplot(y_train)

import numpy as np
# y분포를 log(y)+1로 정규화
y_train= np.log1p(y_train)
y_test= np.log1p(y_test)
sns.distplot(y_train)
##########모델 학습

model = Ridge(alpha=1.0)
model.fit(x_train, y_train)

##########모델 검증
# x 정규화 전후 별차이 없다
print(model.score(x_train, y_train)) #y분포 비정규화 0.883 -> 정규화 0.939 ->0.938
print(model.score(x_test, y_test)) #y분포 비정규화 0.77 -> 정규화 0.898 -> 0.898


#%%
#boston 고차다항회귀
# Polynomial Regression 다항회귀
# 피처(항)들의 고차다항피처 변환시킨 회귀
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

# boston 데이타셋 로드
boston = load_boston()

# boston 데이타셋 DataFrame 변환
bostonDF = pd.DataFrame(boston.data , columns = boston.feature_names)
bostonDF['PRICE'] = boston.target
bostonDF.head()
#%%

#각 컬럼이 회귀결과  주택 가격에 미치는 영향도를 시각화 차트로 표현
# 시본의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현
# 2개의 행과 4개의 열을 가진 subplots를 이용. axs는 4x2개의 ax를 가짐.
#RM(양의 방향성),LSTAT(음의 방향성) 영향도 큼 
fig, axs = plt.subplots(figsize=(16,8) , ncols=4 , nrows=2)
lm_features = ['RM','ZN','INDUS','NOX','AGE','PTRATIO','LSTAT','RAD']
for i , feature in enumerate(lm_features):
    row = int(i/4)
    col = i%4
    # 시본의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현
    sns.regplot(x=feature , y='PRICE',data=bostonDF , ax=axs[row][col])



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as plt

#bostonDF['PRICE']을 y_target 대입 후 bostonDF['PRICE'] 열 삭제
y_target = bostonDF['PRICE']
x_data_df = bostonDF.drop('PRICE',axis=1,inplace=False)

from sklearn.preprocessing import StandardScaler
stand = StandardScaler()
x_data = stand.fit_transform(x_data_df)  # 훈련용 데이터를 표준화한다


#Polynomial Regression 다항회귀
from sklearn.preprocessing import PolynomialFeatures

#만약 단항값x1,x2 라면  2차 다항값 1,x1,x2,x1^2,x1x2,,x2^2로 변환 
#차수가 너무 높으면 과적합 가능성
poly_reg = PolynomialFeatures(degree = 3) #2차(degree는 차수) 다항식 피처로 변환
x_data = poly_reg.fit_transform(x_data)

x_train , x_test , y_train , y_test = train_test_split(x_data , y_target ,test_size=0.3, random_state=1)
#lr = LinearRegression()

from sklearn.linear_model import Ridge
lr = Ridge()
lr.fit(x_train ,y_train )

##########모델 검증
#Polynomial Regression전 학습정확도 0.727
#Polynomial Regression후 학습정확도 
#LinearRegression:학습정확도 0.935
#Ridge: 학습정확도  0.931
print(lr.score(x_train, y_train))

#%%
y_preds = lr.predict(x_test)
mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)
#Polynomial Regression전 MSE : 17.297 , RMSE : 4.159
#LinearRegression: MSE : 15.556 , RMSE : 3.944
#Ridge: MSE : 9.975 , RMSE : 3.158
print('MSE : {0:.3f} , RMSE : {1:.3F}'.format(mse , rmse))

#예측값분산/실측값분산 1에 가까우면 에측정확도 높음
#Polynomial Regression전 0.757
#LinearRegression:  에측정확도 0.781
#Ridge: 에측정확도 0.860
#Polynomial Regression Ridge 모델이Polynomial Regression LinearRegression 보다 성능이 좋음
print(lr.score(x_test, y_test)) 
##Polynomial Regression전 0.757
#LinearRegression:  Variance score : 0.822
print('Variance score : {0:.3f}'.format(r2_score(y_test, y_preds)))

'''
bostonDF = pd.DataFrame(boston.data, columns=boston.feature_names)
import statsmodels.api as sm
bostonCDF  = sm.add_constant(x_train)
model_boston = sm.OLS(y_train, bostonCDF)
result_boston = model_boston.fit()
# R-squared:                       
print(result_boston.summary())

'''

