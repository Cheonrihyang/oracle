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

#%%
'''
다중공선성(Multicolleniarity)
회귀분서에서 피처(독립변수)간의 상관성이 
아주 높은 피처(중복성피처)들은 
오히려 오차증가 성능저하

독립변수들 간에 강한 상관성
을 가지는 다중공선성(Multicolleniarity)가 존재하면 
다른 독립변수에 의존하는 변수를 없애는 것이다. 
가장 의존적인 독립변수를 선택하는 방법으로는 VIF(Variance Inflation Factor)를 사용할 수 있다. 
회귀계수의 분산이 매우 커지게 되면 회귀 불안

변수별 VIF >=5 :  다중공선성 존재
VIF >=10 :  다중공선성 심각
VIF >=10인 독립변수들중에서 
가장 큰 VIF 변수를 제거
-> 나머지 변수들을 대상으로 VIF 계산, 제거의 
과정을 반복 

'''
from statsmodels.stats.outliers_influence import variance_inflation_factor
#X :x데이터
def filter_multicollinearity(X, thresh=5.0):

    from datetime import datetime
    start_tm = datetime.now()

    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        #ix 번 독립변수의 VIF를 계산한다.
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix) 
               for ix in range(X.iloc[:, variables].shape[1])]
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('==> [Dropped variable] : ' , X.iloc[:, variables].columns[maxloc])
            del variables[maxloc]

            if len(variables) > 1:
                dropped = True
    print('[Remaining variables] :')

    print(X.columns[variables])

    print('[Elapsed time] :', str(datetime.now() - start_tm))

    return variables

#%%

y=np.arange(0,11)
x=np.arange(0,6)
import seaborn as sns
sns.displot(y)


#%%
# boston 데이타셋 로드
boston = datasets.load_boston()
bostonDF = pd.DataFrame(boston.data , columns = boston.feature_names)
bostonDF['PRICE'] = boston.target
'''
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

'''
from sklearn.model_selection import train_test_split
#bostonDF['PRICE']을 y_target 대입 후 bostonDF['PRICE'] 열 삭제
#train_test_split 때문에 별도로 PRICE를 추출 y_target에 분리
y_target = bostonDF['PRICE']
x_data = bostonDF.drop('PRICE',axis=1,inplace=False)

#1. 정규화로 다중공선성 낮추기
# 피처 특정값의 분포가 치우친 왜곡(skew)은 표준 혹은 민맥스 혿은 로그 정규화
# 성능향상이 없으면 피처정규화 데이터에 다항 특성 적용 
# 타겟 특정값의 분포가 치우친 왜곡(skew)은 보통 로그 정규화
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing \
import MinMaxScaler
#stand = MinMaxScaler()
stand = StandardScaler()
#x_data = stand.fit_transform(x_data)  # 훈련용 데이터를 표준화한다
stand.fit(x_data)  
#테스트 데이터는 transform 만 하자 
x_data = stand.transform(x_data)  # 훈련용 데이터를 표준화한다# 훈련용 데이터를 표준화한다

#2. 정규화후  VIF 높은 변수의 삭제로 다중공선성 더 낮추기
#설명변수 X DataFrame과 VIF 기준(threshold)을 5로 설정하고, 순차적인 제거
#tax 제거

remained_idx = filter_multicollinearity(pd.DataFrame(x_data))
print('index after filtering multicollinearity:', remained_idx)
x_data =x_data[:,remained_idx]

#train_test_split
x_train , x_test , y_train , y_test = train_test_split(x_data , y_target 
,test_size=0.3, random_state=1)

lr = LinearRegression()
lr.fit(x_train ,y_train )
#예측전 score 확인 score를 R2로 볼 수 있다 
lr.score(x_train,y_train) #0.71 -> 0.70
lr.score(x_test,y_test) #0.78 -> 0.77

# test 데이터로 예측 
y_preds = lr.predict(x_test)
#예측후  rmse, r2확인
mse = metrics.mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)
rmse# 4.45 4.54
metrics.r2_score(y_test, y_preds)#0.78

#statsmodels의 통계 정보
import statsmodels.formula.api as smf
# df = pd.DataFrame(x_data, columns = boston.feature_names )
# f= 'PRICE~'+'+'.join(boston.feature_names)
# df['PRICE'] = boston.target

#vif 용 DataFrame
df = pd.DataFrame(x_data , columns = boston.feature_names[remained_idx])
f= 'PRICE~'+'+'.join(boston.feature_names).replace('+'+boston.feature_names[9], '')
df['PRICE'] = boston.target
model=smf.ols(data=df,formula=f) 
reg_model = model.fit()

#평가
reg_model.summary() 
'''
#조건수 :1.51e+04
# R-squared: 0.741
Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.51e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

정규화 ->
Notes 메세지 사라짐
#조건수 :9.82
R-squared: 0.741
정규화 + vif ->
#조건수 : 5.87
#R-squared: 0.735
'''




