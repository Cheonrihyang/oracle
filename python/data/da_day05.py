# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:52:15 2024

@author: ORC
"""

#%% 시리즈

import pandas as pd
import numpy as np

#판다스 시리즈 배열 선언
obj = pd.Series([4,7,-5,3])
#배열 인덱스 선언
obj2= pd.Series([4,7,-5,3], index=['d','b','a','c'])

#넘파이삽입 가능
obj = pd.Series(np.array([4,7,-5,3]))

#인덱스 확인
obj.index
obj2.index

#인덱스로 슬라이싱 가능
obj2['d':'a']
obj2[['d','a']]
#식별번호로도 슬라이싱 가능
obj2[0:3]
obj2[[0,3]]

#%% 데이터프레임

import pandas as pd
import numpy as np

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
#판다스 데이터프레임 배열 선언
df = pd.DataFrame(data)
df.index

#키값(칼럼명) 이름
df.columns
#벨류값의 배열 출력
df.values

#1
data=[[1,2,3],[4,5,6]]
df=pd.DataFrame(data)
#인덱스명 수정 가능
df.index = ['a','b']
df.columns
#컬럼명 수정 가능
df.columns=['c0','c1','c2']

#2
data=[[1,2,3],[4,5,6]]
#데이터프레임 생성할때 인덱스,컬럼명 설정 가능
df2=pd.DataFrame(data,index=['a','b'],columns=['c0','c2','c3'])

#%% 슬라이싱

import pandas as pd


data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}

#없는 컬럼을 선언하면 디폴트는 null값으로 자동생성
df = pd.DataFrame(data, columns=["year", "names", "points", "penalty"],
index=["one", "two", "three", "four", "five"])

df['year'] #year열 출력
df['names']
df[['year','points']] #year열과 points열 출력
df['one':'three'] #행 연속 슬라이싱은 가능
#df["year":"points"] 같은 방식으로 열 연속 슬라이싱은 불가능함.

#단일행 기반 연속 슬라이싱 loc,iloc
df.loc['one',:] #index로 선택
df.iloc[0,:] #번호로 선택
df.loc['one']
df.loc['one':'three',:] #다중행 슬라이싱
df.iloc[0:3,:]

#단일 인덱싱 [열][행]순서임([행][열]이 아님)
df['names']['two']
df.loc['two','names'] #행기반

df.loc[:,'year':'names']
df.iloc[:,0:2]
df.loc[:,['names','year']]
df.iloc[:,[1,0]]
df.loc['one':'three','year':'names']
df.iloc[0:3,0:2]
df.loc[['one','three'],['year','names']]

#at,iat 단일행 단일열 특정요소 선택. 속도가 빠름
df.at['one','names']
df.iat[0,1]

#%% 조건문

import pandas as pd

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}

df = pd.DataFrame(data, columns=["year", "names", "points", "penalty"],
index=["one", "two", "three", "four", "five"])

#조건에 맞는 배열 반환
bool=df['year']>2014
df.loc[bool,:]

#다중조건
df.loc[(df['points']>2) & (df['points']<3),'points']
df.query('points>2 &points<3')['points']
df.loc[(df['points']==2.9) | (df['points']==2.4),'points']
df.query('points==[2.4,2.9]')["points"]

df.query('year in [2015,2016]') #year에 2015 2016이 포함되는가

p=2
df.query('points>@p & points<3') #쿼리는 조건안에 변수 사용가능 @를 붙임

#%%퀴즈

import pandas as pd

titanic = pd.read_csv('titanic.csv')

#읽어 들인 승객의  5명을 출력
print(titanic.loc[[36,92,168,843,267]])
#승객 5명의  Survived와 Sex, Fare열을 출력
print(titanic.loc[[36,92,168,843,267],['Survived','Sex','Fare']])
#사망자 수를 출력
print(len(titanic.query("Survived==0")))
titanic.query("Survived==0").shape[0]

#Fare가 50 이상인 Fare ,Survived 열을 출력
print(titanic.query("Fare>50")[['Fare','Survived']])
#남자(male)이면서 Pclass가 1인 승객수를 출력
print(len(titanic.query("Sex in 'male' & Pclass==1")))
#남자이면서 Pclass가 1이고 Fare가 50 이상인 승객수를 출력
print(len(titanic.query("Fare>50 & Pclass==1 & Sex in 'male'")))

#%% 데이터추가

import pandas as pd

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)

#행추가

#마지막 행에 배열 삽입
df.loc[df.shape[0]]=['Smith',2013,0]
df.loc[len(df)]=['Smith',2013,4.0]

#df.loc[len(df)]=pd.Series(['Smith',2013,4.0]) 시리즈를 그냥 넣으면 null이 들어감
#시리즈는 이방법으로 삽입
df=df.append(pd.Series(['Smith',2013,4.0],index=df.columns),ignore_index=True)

#df = df.append(['Smith',2013,4.0]) 이상하게 삽입됨 사용X

#열추가
df['penalty'] = [0.1,0.2,0.3,0.4,0.5]
df['penalty2'] = 0
df['penalty3']=np.arange(0.1,0.6,0.1)
df['penalty4']=pd.Series([0.1,0.2,0.3,0.4,0.5],index=df.index)

#%% 데이터수정

import pandas as pd

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)

#행 검색후 값 대입
df.loc[df['points']>3,:]=0

#열 검색후 값 대입
df["penalty"]=[0.2,0.2,0.3,0.4,0.5]

#특정요소 수정
df["penalty"][0]=0.1 #비권장
df.loc[0,'penalty']=0.1

#칼럼 이름 변경
df.columns = ['n','y','p']
#특정 칼럼 이름 변경 rename(columns={키(기존이름):벨류(바뀔이름)})
df=df.rename(columns={'n':'names'})

#조건에 맞는 열수정
import numpy as np
np.where(df['names']=='Kilho')
df['names'].where(df['names']=='Kilho','k')

#%% 문자열 일부 수정

import pandas as pd

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": ['1914', '19 15', '1916', '2015', '2016'],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)

#str.replace를 사용해 19를 20으로 바꾸고 공백문자 제거후 int로 변환
df['year']=df['year'].str.replace('19', '20').str.replace(' ','').astype(np.int32)



#%% 데이터삭제

import pandas as pd

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)

#행삭제

#index까지 삭제됨.
df = df.drop(2)
#index재정의
df = df.reset_index(drop=True)

#동일행 중복삭제
df.loc[len(df)] = ['Smith',2013,4.0]
df.loc[len(df)] = ['Smith',2013,4.0]
df.loc[len(df)] = ['Smith',2013,4.0]
df=df.drop_duplicates()

#열삭제
df.drop('year',axis=1)
#inplace=df 자신에 결과가 적용
df.drop('year',axis=1,inplace=True)
del df['year']

df.drop(['year','points'],axis=1,inplace=True)
df.columns

#%% 구조확인

import pandas as pd

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)

df.info()
#첫2개만 꺼낸다
df.head(2)
#마지막2개만 꺼낸다
df.tail(2)

#인덱스 설정
#중복체크
df.duplicated("points").sum()
#적합한 키를 index로 설정
df.set_index('points',inplace=True)

#설정된 인덱스 열 삭제
df.reset_index(drop=True,inplace=True)
#삭제하지 않고 되돌림
df.reset_index(inplace=True)

#중복값 제거후 배열로 반환
df['points'].unique()
df['year'].unique()

#%% 널값처리

import pandas as pd
import numpy as np

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)

df.loc[0,'points'] = np.nan

#널값인 행 시리즈리턴
df.loc[df['points'].isnull(),:]
#널값인 행 데이터프레임 리턴
df.loc[df['points'].isna(),:]
#널값이 아닌 행 시리즈리턴
df.loc[df['points'].notnull(),:]
#널값이 아닌 행 데이터프레임 리턴
df.loc[df['points'].notna(),:]


#널값 개수 확인
df['points'].isnull().sum()

#null을 0으로 채운다
df['points'].fillna(0)


#행에 널값이 하나라도 있는 경우 행 삭제
df.dropna(how='any')
#행이 모두 널값인 경우 행 삭제
df.dropna(how='all')

#%% 변환

#넘파이 배열로 변환
df.values
df.to_numpy()

#리스트변환
df.to_numpy().tolist()

#딕셔너리 변환
df.to_dict()

#%%

import pandas as pd
import numpy as np

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)

#열마다 합산
df.sum()

#행마다 합산시 문자열은 자동으로 배제
df.sum(axis=1)

#특정행 합산
df['points'].sum()
df.loc[:,'points'].sum()

df.loc[0,:].sum()#에러발생
df.loc[0,'year':'points'].sum()

df.min()
df.mean()

#중위값
df.median()
df.quantile(0.5)

df.max()

df['points'].mean()
df['points'].quantile(0.25)

#4개 영역 동일 개수로 분할
pd.qcut(df['points'],4)

df['points'].var()
df['points'].std()

#사이즈 구하기
df['points'].count() #널값은 제외됨
df['points'].shape[0] #널값 포함

#최빈값
df['year'].mode()


#%% 퀴즈

import pandas as pd
import numpy as np

score={
'이름':['홍길동','이순신','강감찬'],
 '국어':[200,95,87],
 '영어':[90,92,87],
 '수학':[85,88,99],
 }
df=pd.DataFrame(score)

#학생별 점수 합계 및 평균 출력

#단 국어 점수의 이상값(점수가 100보다 크거나 0보다 작다)을 
#np.where() 활용하여 NaN으로 수정후 NaN을 최소값으로 처리
df['국어'] = np.where((df['국어'] > 100) | (df['국어'] < 0), np.nan, df['국어'])
df['국어'].fillna(df['국어'].min(), inplace=True)

df.set_index('이름',inplace=True)
print(df.sum(axis=1))
print(df.mean(axis=1))


#%% 다중 집계함수

import pandas as pd
import numpy as np

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)
df['points'].min()
df['points'].max()

#집계함수 이름를 리스트로 묶어서 삽입
df['points'].agg(['max','min'])
df['points'].aggregate(['max','min']) #이름만 다르고 동일
df['points'].agg(['max','mean','min'])

#다중열 다중집계
df[['year','points']].agg(['max','mean','min'])

#모든열 모든 집계
df.describe()
#특정열 모든 집계
df['points'].describe()
df['names'].describe() #문자는 집계가 다름

#%% 그룹바이

#년도별
dfg=df.groupby('year').mean()
#인덱스 자동설정 안함
dfg=df.groupby('year',as_index=False).mean()
dfg=df.groupby('year',as_index=False).count()

#년도별 포인트의 갯수
dfg=df.groupby('year').count()
dfg['points']
df.groupby('year').count()['points']
df.groupby('year')['points'].count()

#다중그룹
#년도별 이름별 통계
df.groupby(['year','names'])['points']

#다중그룹의 다중집계
df.groupby(['year','names'])['points'].agg(['max','mean','min'])
df.groupby(['year','names'])['points'].describe()


#IQR(Inter-Quartile Range, Q3 - Q1
def iqr(x):
    q3,q1 = x.quantile([0.75, 0.25])
    res = q3 - q1
    return res

df.groupby('year').agg(iqr)

#%% 정렬

import pandas as pd
import numpy as np

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)

#년도별 names,points의 갯수
df.groupby('year').count()
#년도별 행의갯수
df.groupby('year').size()
df['year'].value_counts() #자동 내림정렬
df['year'].value_counts(normalize=True).head(2) #빈도를 퍼센트로 변환(점유율)

df[['year','points']].value_counts(normalize=True)


df=pd.DataFrame(np.random.randn(6,4))
df.columns = ['A','B','C','D']

#열 오른정렬
df.sort_values(by='D')
#열 내림정렬
df.sort_values(by ='D',ascending=False)

#인덱스 내림정렬
df.sort_index(axis=0,ascending=False)

#다중정렬
#C열값이 같은 그룹내에서 D로 2차 내림정렬
df.sort_values(['C','D'],ascending=[True,False])

#%% 퀴즈

#학생별 평균구해서 평균컬럼 추가 후 평균으로 내림정렬후 이름으로 2차오름정렬하여 출력
#단 결측치를 가지는 학생행은 삭제

#-출력화면
#이순신  91.67
#홍길동  91.67
#강감찬  91.00

score={
'이름':['홍길동','이순신','강감찬','갑돌이'],
 '국어':[100,95,87,np.nan],
 '영어':[90,92,87,50],
 '수학':[85,88,99,50],
 }
df = pd.DataFrame(score)

df['평균']=df.mean(axis=1)
df.dropna(how='any',inplace=True)
df.set_index('이름',inplace=True)

print(round(df.sort_values(['평균','이름'],ascending=[False,True])['평균'],2))

#%% 표본 추출

import pandas as pd

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)

df.head(2)
df.tail(2)

#무작위 2개 추출 중복허용X
df.sample(2)
df.sample(2,replace=True) #중복허용

#%% 날짜

import pandas as pd

#2022-12-01 ~ 2022-12-31 객체생성
#시작일,일수
pd.date_range('2022-12-01',periods=31)
pd.date_range('20221201',periods=31)

#시작일,마지막일
d=pd.date_range('2022-12-01',end='2022-12-31')

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"date": ['20140101','20150212','20161027','20151111','20161212'],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)

#형식화된 날자 시리즈
d = pd.to_datetime(df['date'],format='%Y-%m-%d')
df['date2']=d
d = pd.to_datetime(df['date']) #디폴트로 위와 동일한 포멧 적용
df['date3']=d

#dt=datetime접근자
#년도만 추출
df['date2'].dt.year
#월만 추출
df['date2'].dt.month
#일만 추출
df['date2'].dt.day

from datetime import datetime
datetime.strptime('2024-07-18 18:13:00', '%Y-%m-%d %H:%M:%S')
d = pd.to_datetime('2024-07-18 18:13:00')

from datetime import timedelta
d-timedelta(days=3)

#%% 카테고리

import pandas as pd

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)

df.sort_values('year')
df.info()

#문자열을 카테고리로 반환(메모리절약,데이터 분석에 유리)
df['names'] = df['names'].astype('category')

#카테고리 리스트출력
df['names'].cat.categories

#카테고리 우선순위 설정
df['names'].cat.reorder_categories(['Kilho','Charles'],inplace=True)
df['names'].cat.as_ordered()
#카테고리 우선순위대로 정렬
df.sort_values('names')

#카테고리 우선순위대로 값대입
df['names'].cat.categories=[0,1]
df['names'].cat.categories=[]

#year에 20 지우기
df['year']=df['year'].astype("category")
df['year'].cat.categories=[14,15,16]

#카테고리 타입이라 합연산이 불가능
df['year'].sum()

#%% 퀴즈

titanic = pd.read_csv('titanic.csv')

#Pclass 객실등급 Fare :탑승료(운임) Sex 성별 male, female Survived 생존여부 0,1

#평균 탑승료을 출력한다
print(titanic['Fare'].mean(axis=0))

#성별 생존자와 사망자수를 출력한다
print(titanic.groupby(['Sex','Survived'])['Survived'].count())
titanic[['Sex','Survived']].value_counts(normalize=True)

#객실등급별 생존자와 사망자수를 출력한다
print(titanic.groupby(['Pclass','Survived'])['Survived'].count())
titanic[['Pclass','Survived']].value_counts(normalize=True)

#'Fare_Range' 열을 추가후 Fare_Range 열을 다음조건으로 값을 수정한다 
#-조건
#Fare<=7.91 이면 1
#Fare>7.91 이고 Fare<=14.454 이면 2
#Fare>14.454이고Fare<=31 이면 3
#Fare>31이고 Fare<=513 이면 4

#titanic['Fare_Range']=0
#titanic['Fare_Range'] = titanic['Fare_Range'].where(titanic['Fare']>7.91,1)
#titanic['Fare_Range'] = titanic['Fare_Range'].where((titanic['Fare']<=7.91) |
#                                                    (titanic['Fare']>14.454),2)
#titanic['Fare_Range'] = titanic['Fare_Range'].where((titanic['Fare'] <= 14.454) |
#                                                    (titanic['Fare']>31),3)
#titanic['Fare_Range'] = titanic['Fare_Range'].where((titanic['Fare'] <= 31) |
#                                                    (titanic['Fare']>513),4)

titanic['Fare_Range'] = pd.cut(titanic['Fare'], 
                        bins=[0,7.91,14.454,31,513], 
                        labels=[1,2,3,4],include_lowest=True)

#Fare_Range별로 평균 생존율를 출력한다.
print(titanic.groupby('Fare_Range')['Survived'].mean())

#Sex 성별 male, female을  category(범주형)으로 변환하여 코드화한다.
titanic.groupby(titanic['Fare_Range'])['Survived'].mean()
titanic['Sex'] = titanic['Sex'].astype('category')
titanic['Sex'].cat.categories

#%% 포함여부확인

import pandas as pd

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)

df.query('points==[2.4,2.9]')

#isin이 포함된 행을 boolean배열로 반환
df[df['points'].isin([2.4,2.9])]

#%% 사용자 정의함수

df = pd.DataFrame([[1,2,3],[4,5,6]],
                  columns=['A','B','C'])

def two_times(x):
    return x*2

#사용자 정의함수 사용
df['A'].apply(two_times)
df['A'].map(two_times)

#시리즈일 경우 map 사용이 가능하지만 다중열일경우 불가능
df.apply(two_times)
df.map(two_times)

df.apply(two_times,axis=1)


data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)

def max_minus_min(x):
    return x.max() - x.min()
func = lambda x: x.max() - x.min()

df[['year','points']].apply(max_minus_min, axis=0)
df[['year','points']].apply(func, axis=0)

df[['year','points']].groupby('year').apply(max_minus_min)

#%% 조인

import pandas as pd

#고객 테이블(사전{컬럼:[]})
df_left = pd.DataFrame({
    '고객번호': [1001, 1002, 1003, 1004, 1005, 1006, 1007],
    '이름': ['둘리', '도우너', '또치', '길동', '희동', '마이콜', '영희']
})
#주문테이블(사전{컬럼:[]})
df_right = pd.DataFrame({
    '고객번호': [1001, 1001, 1005, 1006, 1008, 1001],
    '금액': [10000, 20000, 15000, 5000, 100000, 30000]
})

#이너조인
pd.merge(df_left,df_right,how='inner',on='고객번호')
pd.merge(df_left,df_right)

#아우터조인
pd.merge(df_left,df_right,how='left',on='고객번호') #left조인
pd.merge(df_left,df_right,how='right',on='고객번호') #right조인
pd.merge(df_left,df_right,how='outer',on='고객번호') #양쪽조인




data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)

#위아래로 붙이기
df.append(df,ignore_index=True)

df1 = pd.DataFrame([{'name':'John','job':'teacher'}
                    ,{'name':'Candy','job':'student'}
                    ,{'name':'Fred','job':'developer'}])

df2 = pd.DataFrame([{'name':'Ed','job':'engineer'}
                    ,{'name':'Jack','job':'farmer'}
                    ,{'name':'James','job':'student'}
                    ,{'name':'James','job':'student'}])

pd.concat([df1,df2],axis=0,ignore_index=True)
pd.concat([df1,df2],axis=1,ignore_index=True)


#%% 퀴즈

import pandas as pd

#1. df과df2를 상하로 합쳐본다 

df = pd.DataFrame({
    '국어': [1, 2, 3],
    '영어': [4, 5, 6]
})

df2=pd.DataFrame(
    {'국어':[1,2,3]
    ,'영어':[4,5,6]    
    })
df=df.append(df2,ignore_index=True)

#2. 학생별 두 과목 두 과목점수 평균 출력한다.(apply 사용)
fun = lambda x : x.mean()

# 각 행에 대해 함수 적용
df[['국어','영어']].apply(fun,axis=1)

#%% csv 쓰기

data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
"year": [2014, 2015, 2016, 2015, 2016],
"points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)

df.to_csv('frame.csv',mode='w',index=True,encoding='utf-8',header=True)
df.to_csv('frame.csv',encoding='utf-8',index=False,header=True)

#%% csv 읽기

df = pd.read_csv('frame.csv',encoding='utf-8')
df = pd.read_csv('frame.csv',encoding='utf-8',header=None) #헤더가 없으면 자동정수설정
df.columns=['names','year','points']


#정규표현식 사용해서 데이터 정리
#/s+ 길이가 정해지지 않는 공백 
pd.read_csv('fname2.csv',sep='\s+',na_values=['$','?'])

#%% 퀴즈

import pandas as pd
import numpy as np

#총인구수, 세대수,남자_인구수,여자_인구수 4개열의(열별) 합계를 출력
#단 특정 열의 ','와 공백문자를 제거

df = pd.read_csv('population_2020.csv',encoding='utf-8')

def func(col):
    return df[col].str.replace(',','').str.replace(' ','').astype(np.int32).sum()
def func2(col):
    return int(col.replace(',','').replace(' ',''))
def func3(col):
    return format(col,',')

for x in ['2020년02월_총인구수','2020년02월_세대수','2020년02월_남자 인구수','2020년02월_여자 인구수']:
    print(f'{x}는 {format(func(x),",")} 입니다.')

#1. 한글컬럼이름이 조금 복잡해서 컬럼이름을 수정 간소화하고 loc[] 방식을 사용한다.
#- 새컬럼이름
#['지역', '총인구수', '세대수', '세대당_인구', '남자_인구수', '여자_인구수', '남여_비율']
df.columns=['지역', '총인구수', '세대수', '세대당_인구', '남자_인구수', '여자_인구수', '남여_비율']

#2. 총인구수, 세대수,남자_인구수,여자_인구수의 합계를 출력하고
#총인구수 순으로 행들을 내림정렬한다.

func('총인구수')
for x in ['총인구수','세대수','남자_인구수','여자_인구수']:
    print(f'{x}는 {format(func(x),",")} 입니다.')
    df[x]=df[x].apply(func2)
    
df.sort_values(by='총인구수',ascending=False,inplace=True)
print(df)

for x in ['총인구수','세대수','남자_인구수','여자_인구수']:
    df[x]=df[x].apply(func3)