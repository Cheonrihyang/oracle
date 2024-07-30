# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:17:57 2024

@author: ORC
"""
#%%

#데이터 수집,평기
#x,y 데이터분할
#이상치 결측치 처리
#x 표준 혹은 인덱스 크기 정규화 스케일링
#피처간 시각적 상관 분석으로 다중공산성 대략적인 파악 과적합 처리
#타겟의 분포가 치우쳤을시 표준or로그 정규화
#데이터분할(k-분할 교차검증)
#회귀 정규화 과적합문제 해소
#회귀학습(파라미터 그리드서치 k-분할 교차검증)
#성능향상이 없으면(선형관계로 설명이 어려울경우) 피처정규화 데이터에 다항다차 특성 적용
#예측 평가할 새 데이터 피처 정규화
#예측 모델 평가

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
                                                                    test_size=0.25,
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
test=boston.data
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


#%%

import pandas as pd
from sklearn.preprocessing import LabelEncoder

grd=['A','C','B','B','A']

le = LabelEncoder()
le.fit(grd)
y_data=le.transform(grd)
le.classes

#역변환
le.inverse_transform(y_data)


#%% 다중컬럼 인코딩 

items=['TV','냉장고','전자렌지','컴퓨터','선풍기']#제품명
grd=['A','C','B','B','A']#제품등급

import pandas as pd
df = pd.DataFrame({'grd':grd,'items':items})
# le.fit(다중컬럼들) 오류
# label column 별로 Label Encoder object 생성
label_column = ['grd','items'] 
label_encode_list = []
label_encoder_list = []
for column_index in label_column:
    le = LabelEncoder()
    le.fit(df[column_index])
    le.transform(df[column_index])
    label_encoder_list.append(le)
    label_encode_list.append(le.transform(df[column_index])) #각 컬럼 별로 label encode 배열 저장

#label_encode_list = [array([0, 2, 1, 1, 0]), array([0, 1, 3, 4, 2])]    
y0=label_encode_list[0]
label_encoder_list[0].inverse_transform(y0)

#%% 원핫인코딩

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

grd=['A','C','B','B','A']
df=pd.DataFrame(grd)
pd.get_dummies(df)


le = LabelEncoder()
le.fit(grd)
y_data=le.transform(grd)

oh=OneHotEncoder()
y_data = y_data.reshape(-1,1)
oh.fit(df)
y_data=oh.transform(df)
y_data.toarray()

#%% get_dummies를 이용한 원핫인코딩

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression

##########데이터 로드

train_df = pd.read_excel('https://github.com/cranberryai/todak_todak_python/blob/master/machine_learning/regression/carprice_E1SUl6b.xlsx?raw=true', sheet_name='train')
test_df = pd.read_excel('https://github.com/cranberryai/todak_todak_python/blob/master/machine_learning/regression/carprice_E1SUl6b.xlsx?raw=true', sheet_name='test')

test_df.info()


y_train_df = train_df['가격']
y_test_df = test_df['가격']
x_train_df = train_df.drop(['가격'], axis=1)
x_test_df = test_df.drop(['가격'], axis=1)

x_train = pd.get_dummies(x_train_df,columns=['종류','연료','변속기'])
x_test = pd.get_dummies(x_test_df,columns=['종류','연료','변속기'])



y_train = y_train_df.to_numpy()
y_test = y_test_df.to_numpy()

model = LinearRegression()

model.fit(x_train,y_train_df)

print(model.score(x_train,y_train))
print(model.score(x_test,y_test))




#%% tranformer을 이용한 원핫인코딩

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

items=['TV','냉장고','전자렌지','냉장고','전자렌지']#제품명
grd=['A','C','B','B','A']#제품등급
price=[10,20,20,20,10]

df = pd.DataFrame({'grd':grd,'items':items,'price':price})
df.info()
from sklearn.compose import make_column_transformer

transformer = make_column_transformer( 
    (OneHotEncoder(), ['grd','items']),
    remainder='passthrough')

#그냥 배열임
transformer.fit_transform(df)
#인코딩된 컬럼명 확인
transformer.get_feature_names()

#DataFrame으로 구성
#3개의 컬럼을 가지는 df가
#7개 컬럼을 가지는 enc_df 변환
df=pd.DataFrame(transformer.fit_transform(df),
             columns=transformer.get_feature_names())


#%% get_dummies를 이용한 원핫인코딩

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression

##########데이터 로드

train_df = pd.read_excel('https://github.com/cranberryai/todak_todak_python/blob/master/machine_learning/regression/carprice_E1SUl6b.xlsx?raw=true', sheet_name='train')
test_df = pd.read_excel('https://github.com/cranberryai/todak_todak_python/blob/master/machine_learning/regression/carprice_E1SUl6b.xlsx?raw=true', sheet_name='test')

test_df.info()


y_train_df = train_df['가격']
y_test_df = test_df['가격']
x_train_df = train_df.drop(['가격'], axis=1)
x_test_df = test_df.drop(['가격'], axis=1)


transformer = make_column_transformer( 
    (OneHotEncoder(), ['종류','연료','변속기']),
    remainder='passthrough')
transformer.fit(x_train_df)
x_train = transformer.transform(x_train_df)
x_test = transformer.transform(x_test_df)


y_train = y_train_df.to_numpy()
y_test = y_test_df.to_numpy()


model = LinearRegression()
model.fit(x_train,y_train)

print(model.score(x_test, y_test))


#예측

x_test = [
    [2016, '대형', 6.8, 159, 25, 'LPG', 0, 2359, 1935, '수동']
]

x_test = transformer.transform(pd.DataFrame(x_test, columns=['년식', '종류', '연비', '마력', '토크', '연료', '하이브리드', '배기량', '중량', '변속기']))

y_predict = model.predict(x_test)
print(y_predict[0]) #1802.160302088625

#%% 퀴즈

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#맨하탄의 주택정보를 이용한 임대료 예측 선형회귀모델을 구현 및 평가한다.
#rent 정보가 임대료이고 target이다.

#* 고려사항
#- id같은 식별자와 object 같은 비숫자열인 특성은 학습에서 제외한다.
#- train_test_split()을 적용한다.
#- 회귀계수를 큰 값 순으로 정렬하여 출력한다.

df = pd.read_csv('data/manhattan.csv',encoding='utf-8')
y_target = df['rent']
x_data = df.drop(['rent','rental_id','neighborhood','borough'],axis=1,inplace=False)

x , x_test , y , y_test = train_test_split(x_data , y_target 
,test_size=0.3, random_state=1)

y = y.to_numpy()
y_test = y_test.to_numpy()

model = LinearRegression()
model.fit(x,y)

print(model.score(x,y))
print(model.score(x_test,y_test))
print(np.sort(model.coef_,axis=0)[::-1])

#%% Ridge 모델

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


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
y_data = np.array([3, 5, 7, 10, 12, 7, 13, 13, 12, 13, 12, 6])

##########데이터 분석

##########데이터 전처리

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777)

##########모델 생성

model = Ridge(alpha=1)

##########모델 학습

model.fit(x_train, y_train)

##########모델 검증

print(model.score(x_train, y_train)) #0.9341840732176964

print(model.score(x_test, y_test)) #0.8483560379440559

##########모델 예측

x_test = np.array([
    [4, 6]
])

y_predict = model.predict(x_test)

print(y_predict[0]) #8.388713611241638

#%% Lasso 모델

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


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
y_data = np.array([3, 5, 7, 10, 12, 7, 13, 13, 12, 13, 12, 6])

##########데이터 분석

##########데이터 전처리

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777)

##########모델 생성

model = Lasso(alpha=10)
#lasso는 규제가 너무 강하면 score값이 0이 나옴.

##########모델 학습

model.fit(x_train, y_train)

##########모델 검증

print(model.score(x_train, y_train)) #0.9341840732176964

print(model.score(x_test, y_test)) #0.8483560379440559

##########모델 예측

x_test = np.array([
    [4, 6]
])

y_predict = model.predict(x_test)

print(y_predict[0]) #8.388713611241638

#%% 과적합 방지 시각화

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


np.random.seed(42)
n_samples = 30
X = np.random.rand(n_samples) * 10 - 5
y = 2 * X**2 + 3 * X + 5 + np.random.randn(n_samples) * 5

# 데이터 시각화
plt.scatter(X, y, color='blue', label='Data points')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# 과적합된 모델 (10차 다항식 회귀)
degree = 10
model_overfit = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model_overfit.fit(X[:, np.newaxis], y)

# 예측 값 시각화
X_test = np.linspace(-5, 5, 100)
y_pred_overfit = model_overfit.predict(X_test[:, np.newaxis])

plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_test, y_pred_overfit, color='red', label=f'Overfitted model (degree={degree})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()



# 규제 적용 모델 (릿지 회귀)
alpha = 1.0  # 규제 강도
model_ridge = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
model_ridge.fit(X[:, np.newaxis], y)

# 예측 값 시각화
y_pred_ridge = model_ridge.predict(X_test[:, np.newaxis])

plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_test, y_pred_overfit, color='red', linestyle='dashed', label=f'Overfitted model (degree={degree})')
plt.plot(X_test, y_pred_ridge, color='green', label=f'Ridge model (alpha={alpha})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

#%% lasso 시각화

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


# 데이터 생성
np.random.seed(42)
n_samples = 30
X = np.random.rand(n_samples) * 10 - 5
y = 2 * X**2 + 3 * X + 5 + np.random.randn(n_samples) * 5

# 과적합된 모델 (10차 다항식 회귀)
degree = 10
model_overfit = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=0))
model_overfit.fit(X[:, np.newaxis], y)

# 예측 값 시각화
X_test = np.linspace(-5, 5, 100)
y_pred_overfit = model_overfit.predict(X_test[:, np.newaxis])

# 규제 적용 모델들
alphas = [0.1, 1.0, 10.0]
models_lasso = [make_pipeline(PolynomialFeatures(degree), Lasso(alpha=a, max_iter=10000)) for a in alphas]
preds_lasso = []

for model in models_lasso:
    model.fit(X[:, np.newaxis], y)
    preds_lasso.append(model.predict(X_test[:, np.newaxis]))

# 시각화
plt.figure(figsize=(12, 8))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_test, y_pred_overfit, color='red', linestyle='dashed', label='Overfitted model (alpha=0)')
for i, alpha in enumerate(alphas):
    plt.plot(X_test, preds_lasso[i], label=f'Lasso model (alpha={alpha})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# 모델 성능 평가
for i, alpha in enumerate(alphas):
    score = models_lasso[i].score(X[:, np.newaxis], y)
    print(f'Lasso model (alpha={alpha}) - Score: {score:.4f}')
    mse = mean_squared_error(y, models_lasso[i].predict(X[:, np.newaxis]))
    print(f'Lasso model (alpha={alpha}) - Mean Squared Error: {mse:.4f}')
    
#%% ridge 시각화

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 데이터 생성
np.random.seed(42)
n_samples = 30
X = np.random.rand(n_samples) * 10 - 5
y = 2 * X**2 + 3 * X + 5 + np.random.randn(n_samples) * 5

# 과적합된 모델 (10차 다항식 회귀)
degree = 10
model_overfit = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0))
model_overfit.fit(X[:, np.newaxis], y)

# 예측 값 시각화
X_test = np.linspace(-5, 5, 100)
y_pred_overfit = model_overfit.predict(X_test[:, np.newaxis])

# 규제 적용 모델들
alphas = [0.1, 1.0, 10.0]
models_ridge = [make_pipeline(PolynomialFeatures(degree), Ridge(alpha=a)) for a in alphas]
preds_ridge = []

for model in models_ridge:
    model.fit(X[:, np.newaxis], y)
    preds_ridge.append(model.predict(X_test[:, np.newaxis]))

# 시각화
plt.figure(figsize=(12, 8))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_test, y_pred_overfit, color='red', linestyle='dashed', label='Overfitted model (alpha=0)')
for i, alpha in enumerate(alphas):
    plt.plot(X_test, preds_ridge[i], label=f'Ridge model (alpha={alpha})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# 모델 성능 평가
for i, alpha in enumerate(alphas):
    score = models_ridge[i].score(X[:, np.newaxis], y)
    print(f'Ridge model (alpha={alpha}) - Score: {score:.4f}')
    mse = mean_squared_error(y, models_ridge[i].predict(X[:, np.newaxis]))
    print(f'Ridge model (alpha={alpha}) - Mean Squared Error: {mse:.4f}')
    
#%% k겹 교차검증 cross_var_score

from sklearn.model_selection import cross_val_score

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 붓꽃 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 로지스틱 회귀 모델 생성
model = LogisticRegression(max_iter=200)

# 5-폴드 교차 검증
scores = cross_val_score(model, X, y, cv=5)

# 교차 검증 결과 출력
print("Cross-validation scores:", scores)
print("Average cross-validation score:", np.mean(scores))

#%%

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import KFold
#from sklearn.metrics import r2_score
#from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np

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
y_data = np.array([3, 5, 7, 10 ,12, 7, 13, 13, 12, 13, 12, 6])

##########데이터 분석

##########데이터 전처리

#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777)

##########모델 생성

model = LinearRegression()

##########모델 학습

##########모델 검증

cv_results = cross_validate(model, x_data, y_data) #내부적으로 fit
#cv_results = cross_validate(model, x_data, y_data, cv=5, scoring='r2')
#cv_results = cross_validate(model, x_data, y_data, cv=KFold(n_splits=5), scoring='r2')
#cv_results = cross_validate(model, x_data, y_data, cv=5, scoring=make_scorer(r2_score))
#cv_results = cross_validate(model, x_data, y_data, cv=KFold(n_splits=5), scoring=make_scorer(r2_score))

print(cv_results['test_score'].mean()) #0.9333333333333332
df = pd.DataFrame(cv_results)
df = df.sort_values(by='test_score', ascending=False)
print(df)
'''
   fit_time  score_time  test_score
4  0.000741    0.000837    0.960204
1  0.000488    0.000758    0.568472
2  0.000485    0.000577    0.000000
0  0.000674    0.000684   -1.147559
3  0.000477    0.000811  -11.774764
'''

##########모델 예측

model.fit(x_data, y_data)

x_test = np.array([
    [4, 6]
])

y_predict = model.predict(x_test)

print(y_predict[0]) #1802.160302088625

#%% GridSearchCV

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
#from sklearn.metrics import r2_score
#from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import pandas as pd

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
y_data = np.array([3, 5, 7, 10 ,12, 7, 13, 13, 12, 13, 12, 6])

##########데이터 분석

##########데이터 전처리

#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777)

##########모델 생성

model = Lasso()

##########모델 학습

##########모델 검증

print(model.get_params().keys()) #

param_grid = {
    'alpha': [0.5, 1, 1.5]
}
grid_search = GridSearchCV(model, param_grid=param_grid) 
#grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='r2') 
#grid_search = GridSearchCV(model, param_grid=param_grid, cv=KFold(n_splits=5), scoring='r2') 
#grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring=make_scorer(r2_score))
#grid_search = GridSearchCV(model, param_grid=param_grid, cv=KFold(n_splits=5), scoring=make_scorer(r2_score))

grid_search.fit(x_data, y_data)

print(grid_search.best_params_) #{'alpha': 0.5}
print(grid_search.best_score_) #-2.8938345053645973
df = pd.DataFrame(grid_search.cv_results_)
df = df.sort_values(by='mean_test_score', ascending=False)
print(df[['params', 'mean_test_score']])   
'''
           params  mean_test_score
0  {'alpha': 0.5}        -2.893835
1    {'alpha': 1}        -3.618589
2  {'alpha': 1.5}        -4.449471
'''

##########모델 예측

x_test = np.array([
    [4, 6]
])

best_model = grid_search.best_estimator_
y_predict = best_model.predict(x_test)

print(y_predict[0]) #8.279504382440336

#%% GridSearchCV

df = pd.read_csv('data/manhattan.csv',encoding='utf-8')
y_target = df['rent']
x_data = df.drop(['rent','rental_id','neighborhood','borough'],axis=1,inplace=False)

x , x_test , y , y_test = train_test_split(x_data , y_target 
,test_size=0.3, random_state=1)

param_grid = {
    'alpha': [0.1, 1, 10, 100]
}

model = Ridge()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')


grid_search.fit(x,y)

# 최적의 하이퍼파라미터 출력
print("Best hyperparameters:", grid_search.best_params_)

# 최적의 모델 출력
best_model = grid_search.best_estimator_
print("Best model:", best_model)

# 테스트 세트에서 최적 모델 평가
test_score = best_model.score(x_test, y_test)
print("Test set score with best model:", test_score)


#%% 모델링 예시

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

desc = x_train_df.describe()

#Index(['년식', '종류', '연비', '마력', '토크', '연료', '하이브리드', '배기량', '중량', '변속기'], dtype='object')
print(x_train_df.columns) 


# StandardScaler객체 생성
#scaler = StandardScaler()
#연비, 마력, 배기량의 평균과 분산의 큰차이-> 피처 크기 정규화 필요
from sklearn.preprocessing import StandardScaler
# transformer = make_column_transformer(
#     (StandardScaler(),['연비','마력','배기량']),
#     (OneHotEncoder(), ['종류', '연료', '변속기']),
#     remainder='passthrough')
#make_column_transformer((scaler,[숫자형컬럼들]),
#                         encoder,[범주형컬럼들])
transformer = make_column_transformer(
    (StandardScaler(),['년식','연비','마력','토크','하이브리드','배기량','중량']),
    (OneHotEncoder(), ['종류', '연료', '변속기'])
    )


transformer.fit(x_train_df)
x_train = transformer.transform(x_train_df) #트랜스포머의 transform() 함수는 결과를 넘파이 배열로 리턴
x_test = transformer.transform(x_test_df)

y_train = y_train_df.to_numpy()
y_test = y_test_df.to_numpy()


import seaborn as sns
#평균이 왼쪽으로 치우진 왜곡된 멱함수분포(편포=skew )
#왜도(치우침(편향된 분포)정도 = skewness) > 0 
#오른쪽으로 편포 왜도 < 0    
sns.distplot(y_train)
#로그 변환
import numpy as np
y_train= np.log1p(y_train) 
y_test= np.log1p(y_test)

##########모델 생성

#model = Ridge(alpha=10)
model = Ridge(alpha=1) #피처값이 작아지므로 이 예제에서는 이 경우가 좀 더 점수가 높다
##########모델 학습

model.fit(x_train, y_train)

##########모델 검증
print(model.score(x_train, y_train)) 
print(model.score(x_test, y_test)) 

#%% IQR를 사용한 이상치 제거

from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.linear_model import Ridge
dataset = datasets.load_boston()
x_data = dataset.data
y_data = dataset.target

import seaborn as sns
#이상치 확인
sns.boxplot(data = x_data) 
x_data = pd.DataFrame(x_data,columns=(dataset.feature_names))
#1Q
quartile_1 = x_data.quantile(0.25)
#3Q
quartile_3 = x_data.quantile(0.75)
#IQR =3Q-1Q
IQR = quartile_3 - quartile_1

#1Q- IQR*1.5(ㅗ)보다 작거나 
#3Q+ IQR*1.5(T)보다 큰 데이터는 이상치(outlier)
#이상치 제거
condition = (x_data < (quartile_1 - 1.5 * IQR)) | (x_data > (quartile_3 + 1.5 * IQR)) 
condition.head(50)

#데이터프레임.any(axis=1): 특정값을 가지는 행이 있는지 파악 가능
#x의 특정 피처의 (가로방향으로) 값이 하나라도 True이면
#True를 출력(해에 True가 한개 이상은 존재한다) 
#예로  False,True -> True)
#all() : (모든 피처가 True이면)True를 출력
condition = condition.any(axis=1) #이상한 행 조건
type(condition) #Series

#불리언 값 not
#not True
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
#인덱스번호가 비연속적
x_data = x_data[condition]
y_data= y_data[condition]

#이상치 감소 확인
sns.boxplot(data = x_data)


#%% IQR를 사용한 이상치 제거 더 쉬운 예시

import seaborn as sns
import numpy as np
import pandas as pd

# 간단한 데이터셋 생성 (이상치 포함)
data = pd.DataFrame({
    'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, -10]  # 30과 -10은 이상치
})

# 박스플롯 시각화 (이상치 포함)
sns.boxplot(data=data)
plt.title('Boxplot with Outliers')
plt.show()

# IQR을 사용하여 이상치 제거
Q1 = data['values'].quantile(0.25)
Q3 = data['values'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치 제거
data_clean = data[(data['values'] >= lower_bound) & (data['values'] <= upper_bound)]

# 이상치 제거 후 박스플롯 시각화
sns.boxplot(data=data_clean)
plt.title('Boxplot without Outliers')
plt.show()

# 결과 출력
print("Original data shape:", data.shape)
print("Cleaned data shape:", data_clean.shape)
print("Removed outliers:")
print(data[~((data['values'] >= lower_bound) & (data['values'] <= upper_bound))])