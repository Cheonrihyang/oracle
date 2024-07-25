# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:52:34 2024

@author: ORC
"""

#%% seaborn

# import seaborn as sns
# sns.함수(x=x열이름,y=y열이름,data=데이터셋) 
# sns.함수(x=데이터셋[x열이름],y=데이터셋[y열이름]) 
# barplot : x의 y값들 평균
# boxplot: x의 y 사분위분포

# lineplot
# regplot:  x의 y의  선형 상관 관계(선) (x가 연속형 수열,시계열 ,hue 미지원)
# lmplot:  x의 y의  선형 상관 관계(선) (x가 이산형(범주형,hue 지원)):hue 옵션을 사용해서 카테고리별 비교가 가능

# scatterplot: x의 y의 스캐터 플롯(산점도)
# pairplot: 각 열의 값 조합의 스캐터 플롯
# countplot: x별 빈도수
# histplot
# distplot: x구간별 빈도수 분포 히스토그램,밀도함수

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Malgun Gothic')

tips = sns.load_dataset("tips")#팁 데이터
tips.head()
tips.info()
tips.describe()

#%% barplot
#x별 y 분석
#중심선(오차막대)은 신뢰구간을 의미 
#sns.barplot(x=tips["day"], y=tips["total_bill"])
sns.barplot(x="day", y="total_bill",data=tips)
plt.title("요일별 전체 식사대금")

sns.barplot(x="day", y="total_bill",hue='sex', data=tips)
plt.title("요일별 성별 전체 식사대금")

sns.barplot(x="day", y="tip",data=tips)
plt.title("요일별 팁금액")
#estimator=중간값,오차막대=표준편차
sns.barplot(x="day", y="total_bill",data=tips,
            estimator=np.median,
            ci="sd")
plt.title("요일별 전체 식사대금")
#ci=None하면 오차막대 비출력
sns.barplot(x="day", y="total_bill",data=tips,ci=None)
plt.title("요일별 전체 식사대금")

#%% box(상자)plot
#사분위수와 함께 바깥의 다이아몬드점은 아웃라이어(outlier)
sns.boxplot(x="day", y="total_bill",data=tips)
plt.title("요일별 전체 식사대금")

plt.figure(figsize=(9,7))
sns.boxplot(x="day", y="total_bill",hue='sex', data=tips)
plt.title("요일별 성별 전체 식사대금")

plt.figure(figsize=(9,7))
sns.boxplot(x="day", y="total_bill",hue='smoker', data=tips)

plt.title("요일별 흡연여부별 전체 식사대금")

#인원수 3 경우 : 팁금액이 매우 큰 금액(10)이 보이고
#2Q(중앙값) 조금 위쪽에 보인다
#인원수 4 경우 : 팁금액의 분포가 넓게 다양하게 보인다
plt.figure(figsize=(9,7))
sns.boxplot(x="size", y="tip", data=tips)
plt.title("인원수별 팁금액")
#%% lineplot
# 평균값을 실선으로 95% 신뢰구간을 범위로 시각화 
# x에 따른 y의 변동량(변화량)
# 데이터가 모두 연속적인 실수값이고 주로 x는 시간데이터
sns.lineplot(x="day", y="total_bill",data=tips)
plt.title("요일별 전체 식사대금")
#인원수가 증가하면 팁 증가하는 추이
#인원수가 증가하면 팁의 분포범위(편차) 크다
#예 2인은 분포범위(편차) 일정하고 
#5인은 분포범위(편차) 크다 
sns.lineplot(x="size", y="tip", data=tips)
plt.title("인원수에 따른 팁금액의 변동량")

sns.lineplot(x="total_bill", y="tip",  data=tips)
plt.title("식사대금에 따른 팁금액의 변동량")

#hue 그룹화
sns.lineplot(x="total_bill", y="tip",hue='smoker',  data=tips)
plt.title("흡연여부별 식사대금에 따른 팁금액의 변동량")
#%%
#등고선 스타일
#면적이 좁으면 경사가 가파르고 두 열값간의 상관관계가 높다 
sns.kdeplot(x="total_bill", y="tip",data=tips)
plt.title("흡연여부별 식사대금에 따른 팁금액의 변동량")

#%% lmplot,regplot
#스캐터 + 회귀분석 표현
#total_bill과 tip 간 선형관계(스캐터 + 회귀직선)
sns.regplot(x="total_bill", y="tip",data=tips)

sns.regplot(x=tips["total_bill"], y=tips["tip"])

plt.title("식사대금에 따른 팁금액의 회귀 분포 변동량")

#lmplot은 hue 그룹화 지원
#total_bill과 tip 간 선형관계(스캐터 + 회귀직선)
sns.lmplot(x="total_bill", y="tip", hue='smoker',data=tips)
plt.title("식사대금에 따른 팁금액의 회귀 분포 변동량")

# row='sex',col='time' : 성별, 시간대(점심,저녁)별로 나누어서 그래프 4개 subplot으로 2행 2열로 나타남
sns.lmplot(x="total_bill", y="tip",
           hue='smoker',
           row='sex',col='time',
           data=tips)

#%%  scatter plot
# x의 y의 상관 관계 확인 
sns.scatterplot(x="total_bill", y="tip",data=tips)
plt.title("식사대금에 따른 팁금액")

sns.scatterplot(x="total_bill", y="tip", hue='sex',data=tips)
plt.title("성별 식사대금에 따른 팁금액")

#버블차트(값(버블)의 크기를 다르게하여 산점) 
sns.scatterplot(x="total_bill", y="tip", hue='sex',data=tips,
                size = "tip",legend = True, sizes = (20, 400),
                alpha=0.5)

plt.title("성별 식사대금에 따른 팁금액")

#%% pairplot
# x의 y의 상관 관계 확인하기 위한 기초플롯으로 널리 사용 
#각 숫자형 컬럼(열)들의 모든 상관 관계를 동시에 출력
#모든 열간 값들의 조합 산점도(상관행렬)
##스캐터 + 히스토그램
sns.pairplot(tips) 
#회귀선 포함
sns.pairplot(tips,kind='reg')

#'tip','total_bill' 열의 상관 관계 출력
sns.pairplot(tips,vars=['tip','total_bill'])

#%% 히스토그램(도수분포도)
#x축은 구간계급,  y축은 구간도수
#구간별 도수
sns.histplot(x="total_bill",data=tips)
plt.title("식사대금 도수분포")

#5개 구간
sns.histplot(x="total_bill",data=tips,bins=5)
plt.title("식사대금 도수분포")

#커널 밀도(kernel density) 추정 적용
#kde 곡선이 왼쪽으로 치우친 모양을 확인
sns.histplot(x="total_bill",data=tips,kde=True)
plt.title("식사대금 도수분포")

#%% distplot(분포플롯)
# y축은 커널 밀도(kernel density) 추정으로 정규화된 수치 
sns.distplot(x="total_bill",data=tips) #오류
sns.distplot(x=tips["total_bill"])
sns.distplot(x=tips.loc[:,"total_bill"])
plt.title("식사대금 밀도분포")

#5개 구간(막대)
sns.distplot(x=tips["total_bill"],bins=5)
#%% countplot
# 값(보통 카테고리 값)별 개수
sns.countplot(x="sex", data=tips)
sns.countplot(x="sex",hue='smoker',  data=tips)
#성별 비흡연자수 
tips['smoker'].unique()
sns.countplot(x="sex",  data=tips[tips['smoker']=='No'])

#%% titanic
titanic = sns.load_dataset("titanic") 
titanic.info()
'''
survived 생존여부(수치) 1,0
alive 생존여부(문자) yes,no
pclass 티켓의 클래스,객실등급 1=1st, 2=2nd, 3=3rd 범주(카테고리)형 수치
class 티켓의 클래스,객실등급 범주(카테고리)형 문자
sex 성별 male, female 
age 나이 연속형 수치
sibSp 함께 탑승한 형제와 배우자의 수 
parch 함께 탑승한 부모, 아이의 수  정량형 수치
ticket 티켓 번호 
fare 탑승료 
cabin 객실 번호 
embared 탑승 항구 C=Cherbourg Q=Queenstown S=Southampton 
'''
titanic.head()
titanic.tail()
#무작위 랜덤한 5개 행 추출
titanic.sample(5)
#총 행개수(널값 제외)
titanic['age'].count()
len(titanic['age'])
len(titanic)
len(titanic.index)
titanic.shape[0]
titanic.size #총 요소개수

#class 값별 나머지 열의 값개수 
titanic.groupby('class').count()

#class 값별 개수 
titanic.groupby('class').size()
titanic['class'].value_counts()

# NaN 개수
len(titanic['age']) - titanic['age'].count()
titanic['age'].isnull().sum()
#행 내 값들 중에 NaN이 하나라도 포함되어 있는 경우 해당 행 삭제
titanic.dropna(how='any')

sns.countplot(x="age", data=titanic)


#중복행 개수
titanic.duplicated().sum()
#중복행 제거
titanic.drop_duplicates()

titanic['age'] = titanic['age'].dropna()

#%% quiz
# 전체승객을 나이별로 히스토그램 분포로 출력(distplot)
sns.distplot(titanic.dropna()["age"])

# 전체요금 분포를 히스토그램과 커널밀도 출력(널값은 평균값으로 변환) 
titanic['fare'].fillna(titanic['fare'].mean(), inplace=True)
sns.histplot(titanic['fare'])
sns.distplot(titanic['fare'])

# 남.여 승객수를 출력 (countplot)
sns.countplot(x = "sex", data = titanic)
sns.countplot(titanic["sex"])

# 객실별 승객수를 출력 (countplot)
sns.countplot(x = "class", data = titanic)

titanic.isnull()
titanic['age'].isnull().sum()
sns.heatmap(titanic.isnull(),cbar=False)

#객실(등급)별 생존자와 사망자수(생존/사망 별로 개수로 해석)
#3등석이 사망자 많다
sns.countplot(x='class' ,hue='alive' ,data=titanic)
sns.lineplot(x='pclass' ,y='survived' ,data=titanic)

# 객실별 사망자 수를 출력 (countplot)
sns.countplot(x = "class", 
              data = titanic[titanic["survived"] == 0])
plt.title('Dead of Titanic by Class')
plt.xlabel("Class Level")
plt.ylabel("Count")

#리턴값 axes
ax = sns.countplot(x = "class", 
              data = titanic[titanic["survived"] == 0])
ax.set_title('Dead of Titanic by Class')
plt.set_xlabel("Class Level")
plt.set_ylabel("Count")

# 사망자와 생존자를 출력 (plot.pie)
#as_index = False : 원래 그룹 컬럼이 행인덱스로 되는데  
#그룹 컬럼을 일반컬럼으로 구성하고  행인덱스는 정수인덱스로 생성 
titanic_survived = titanic.groupby(["survived"], 
                   as_index = False).count()["pclass"]
#0    549
#1    342
plt.pie(titanic_survived, labels=["Dead", "Survived"], 
        autopct='%1.1f%%', shadow=True, startangle=90)

#열.value_counts().plot.pie
titanic['survived'].value_counts().plot.pie(explode=[0, 0.1],
                                            autopct='%1.1f%%',
                                            shadow=True)
titanic['alive'].value_counts().plot.pie(explode=[0, 0.1], 
                                         autopct='%1.1f%%',
                                           shadow=True)
# 남/여 성별 사망자와 생존자를 출력  (countplot)
# 남성이 사망자 많다
sns.countplot(x = "survived", hue="sex", data = titanic)

# 객실등급별 나이분포를 출력 (boxplot)
sns.boxplot(x = "pclass", y='age', data = titanic)

# 객실등급별 나이를 출력 (pointplot)
# 선+산(마커)점+신뢰구간
sns.pointplot(x='pclass', y='age', data=titanic)
#%%
# 히트맵
# 자료의 집계 결과를 색깔을 다르게 해서 2차원으로 시각화하는 히트맵
titanic.corr()
sns.heatmap(titanic.corr(),  annot=True)
#객실별(클래스별) 성별 구분 요금 테이블(pivot_table)
#행은 객실 ,열은 성 , 행열값은 요금평균, 집계는 평균
titanic_pt = titanic.pivot_table(index="pclass", 
                                 columns="sex",
                                 values='fare')
plt.figure(figsize=(20,20))
sns.heatmap(titanic_pt ,annot=True)
#fmt='d' 정수
#fmt='.2f' 소수점 이하 2자리수 실수
sns.heatmap(titanic_pt ,fmt='.2f',annot=True)

# 객실별(행이 pclass별)
# 성별(열이 sex) 
# 구분 승객수 테이블(pivot_table) 히트맵
# 요금의 개수(행열값)가 널값 없으므로 승객수이다 
# 집계는 개수 np.size,"size"
import numpy as np
titanic_pt = titanic.pivot_table(index="pclass", 
                                 columns="sex",
                                 values='fare',
                                 aggfunc="size")

sns.heatmap(titanic_pt ,annot=True,fmt='d')

#pointplot: 선+마커플롯(시간의 흐름에 따른 추이)
#나이에따른(나이별) 요금을 출력
plt.figure(figsize=(20,4))
sns.pointplot(x='age', y='fare', data=titanic)

#%% quiz
# 미국 yob(year of birth)별 신생아 이름을 탐색적 데이터 분석한다.
# 데이터셋 : 연도별 이름,성별,이름개수가 있는 파일들의 묶음
#       gender       name   count
# 10250      F       Emma  177410
# 56942      M    William  145893

#전처리 : 각 파일에 상응하는 각 데이터프레임의 통합DF로 병합
#yob201로 시작하는 파일명을 모두 찾아서 읽어들인다.
#특정 디렉터리에 있는 파일 이름 모두를 알아야 할 때  glob 모듈 사용

import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 2010~2018까지 출생한 전체 남녀 수의 합을 구하기
#파일경로 표현식 "data/names/yob201*"
names_data = glob.glob("data/names/yob201*")

names=pd.DataFrame() # 통합DF

for fname in names_data:
    df= pd.read_table(fname,header=None
                      ,names=['name','gender','count'],sep=',')
    df['year'] =fname[14:18]#연도4자리 'data/names\\yob2010.txt'
    #년도별 DataFrame리스트를 통합DF에 추가 합치기 
    names=pd.concat([names,df],ignore_index=True)

names.info()
names.head()


# 2010~2018까지 출생한 전체 남녀 수의 합 
names['count'].sum()

# 2. 년도(x)별 성별(hue)로 구분하여 출생(카운트 y) 그래프를 그리기(sns.barplot())
sns.barplot(x='year',y='count',hue='gender',data=names)

ax=sns.pointplot(x='year',y='count',hue='gender',data=names)
ax.set_title('Count of Birth')  

# 3. 이름의 성별에 따른 빈도수가 가장 높은 이름 10개만 출력해보기(pivot_table 사용)
#aggfunc=np.sum 가능
name_pivot=names.pivot_table(index='name',
                             columns='gender',
                             values='count' ,
                             aggfunc='sum',
                             fill_value=0 )
name_pivot.head()
name_pivot.sort_values(by='F',ascending=False).head(10)
name_pivot.sort_values(by='M',ascending=False).head(10)
#문자이름 상위 10 리스트 추출
name_pivot.sort_values(by='F',ascending=False).head(10).index.tolist()
name_pivot.sort_values(by='M',ascending=False).head(10).index.tolist()

#%% groupby 기반 집계
#이름의 성별에 따른
names.groupby(['name','gender'],as_index=False)
#이름의 성별에 따른 빈도수
name_sum=names.groupby(['name','gender'],as_index=False)['count'].sum()
name_sort=name_sum.sort_values(by='count',ascending=False) 

#문자이름 상위 10 리스트 추출
name_sort[name_sort['gender']=='F'].head(10)['name'].tolist()
name_sort[name_sort['gender']=='M'].head(10)['name'].tolist()

#%% 2018년 평일 2호선 5-6시 사이 정류장별 승차 수 top20 구하고 시각화한다.

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
plt.rc('font', family='Malgun Gothic')

#파일의 헤더줄 1번
df = pd.read_excel('data/서울교통공사_관할역별_일별_시간대별_이용인원_20181231(1-8호선).xlsx',header=1)
df.info()

#1. select 선택열
#구분열은 '평','휴'로 구성
#'호선
#5-6
#역명
#두번째 구분열은 승차 하차
#sum(5-6시)
#2. from df
#3. where 행 필터링 조건 
#평일 2호선  승차
#4. group by  역명
#역명(정류장)별

#열선택 행필터링
df_2ho = df[ (df['구분'] == '평') & (df['호선'] == '2호선') &
            (df['구분.1'] == '승차')]
#널개수 0
#df_2ho['구분.1'].isna().sum()
df_2ho['구분.1'].isnull().sum()

#4. group by 역명
#역명별
#역별 총 승차 인원수(모든 날의 승차 인원수)
#df_2ho.groupby('역명').size() X
df_2ho_total = df_2ho.groupby('역명').sum()['05 ~ 06']
df_2ho_total = df_2ho.groupby('역명')['05 ~ 06'].sum()
type(df_2ho_total) # Series

#TypeError: sort_values() got an unexpected keyword argument
#Type이 Series 이고 sort_values() 단일컬럼이므로 'by' 사용 X
#df_2ho_total.sort_values(by='05 ~ 06', ascending=False).head(20)
df_2ho_top20 = df_2ho_total.sort_values(ascending=False).head(20)


#Series -> DataFrame
#sort_values(by) 사용가능
#df_2ho.groupby('역명')['05 ~ 06'].sum().to_frame()
#df_2ho_total = pd.DataFrame(df_2ho.groupby('역명')['05 ~ 06'].sum())

#'역명'(인덱스)으로 df_2ho_top20 정렬
df_2ho_top20.sort_index(ascending=False)
#%%
#Series 플롯
#ValueError
#sns.barplot(x=df_2ho_top20.index,  y='05 ~ 06',   data=df_2ho_top20) 

#Series인경우 단일컬럼이므로 columns 컬럼인덱스 X 값들이 있다
df_2ho_top20.columns #AttributeError
df_2ho_top20.values

plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(16,10))
#sns.barplot(x=df_2ho_top20.index,y=df_2ho_top20.values) 
#역명 가로 겹친다
#plt.xticks(rotation=90)

#수평(가로)막대
sns.barplot(y=df_2ho_top20.index,
            x=df_2ho_top20.values) 

plt.savefig('이용인원.png')
#%%
2호선이면서 역명이 시청이인 1월 승차수의 합계를  시각화한다.
단 1월의 일자별로 승차수의 합계 추이를 출력

mask1 = (df['역명'] == '시청') & 
df['호선'] == '2호선') & 
(df['구분.1'] == '승차')

df2 = df[ mask1 ]

mask2 = (df2['날짜'] >= '2018-01-01') & 
(df2['날짜'] <= '2018-01-31')
df2 = df2[ mask2 ]

plt.figure(figsize=(16,10)) 
plt.xticks(rotation=90)        
sns.pointplot(x='날짜',y='합 계', data=df2) 

#날짜 'yyyy-mm-dd' 형식화
from datetime import datetime

datetime.strftime(datetime.now(), '%Y-%m-%d')
#datetime.strftime(df2['날짜'], '%Y-%m-%d')
#dt: 해당컬럼 datetime 접근자로 모든 datetime 값들 
df2['날짜'] = df2['날짜'].dt.strftime('%Y-%m-%d')
plt.figure(figsize=(16,10)) 
plt.xticks(rotation=90)        
sns.pointplot(x='날짜',y='합 계', data=df2) 
df2.columns[6:-1]

#%% 2018년 평일이고 2호선 정류장별 18-19 사이 혼잡도의 Top-10
# 2018년 평일 2호선 18-19 사이 정류장별 혼잡도의 Top-10을 분석하여
# 바플롯
# 단 혼잡도는 시간대별 승,하차 이용객수 합을 기반한다. 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#파일의 헤더줄 1번
df = pd.read_excel('서울교통공사_관할역별_일별_시간대별_이용인원_20181231(1-8호선).xlsx',header=1)
df.info()
df.sample(5)
df.describe()
sns.pairplot(data=df, vars=['역번호','05 ~ 06'])

#평일 2호선
mask_gubun =  df['구분']=='평'
mask_line =  df['호선']=='2호선'

df_2ho = df[mask_gubun & mask_line]
#승,하차 이용객수 합
df_2ho_total = df_2ho.groupby('역명')['18 ~ 19'].sum().to_frame()
df_2ho_total.info()
df_2ho_total.head()
#혼잡도의 Top-10
df_2ho_total_top10 = df_2ho_total.sort_values(by='18 ~ 19',ascending=False).head(10)

#차트 출력
plt.rc('font',family='Malgun Gothic')
plt.figure(figsize=(16,10))
plt.xticks(rotation=90)

sns.barplot(y=df_2ho_total_top10.index,
            x=df_2ho_total_top10['18 ~ 19']
            )

#%% word cloud(tag cloud)
# 문서에 언급된 단어들의 빈도수를 파악해서 
# 빈도수가 높은 단어(키워드)일 수록 크게 빈도수가 낮은 단어일 수록 작게 표현하는 시각화 기법
# tag cloud에서는  tag가 키워드
text=open('data/wctest.txt',encoding='utf-8').read()
from wordcloud import WordCloud

#텍스트인경우
# stopwords= 불용어 리스트(빈도수 1)
wordcloud = WordCloud(font_path='c:/windows/fonts/malgun.ttf',
          background_color='white',
          stopwords=['analysis','big']).generate(text)

#리스트인경우 문자열로 변환
text = ' '.join(['워드클라우드','워드클라우드','python','big'])
# stopwords= 불용어 리스트(빈도수 1)
wordcloud = WordCloud(font_path='c:/windows/fonts/malgun.ttf',
          background_color='white',
          stopwords=['analysis','big']).generate(text)

import matplotlib.pyplot as plt
plt.figure(figsize=(22,22)) #이미지 사이즈
#wordcloud 이미지 출력
#이미지의 경계선 계단현상을 보간(interpolation)해서 부드럽게 표현
plt.imshow(wordcloud, interpolation='lanczos') 
plt.axis('off') #x, y 축 숫자 제거

#파일저장1
plt.savefig('wordcloud.png')
#파일저장2
wordcloud = WordCloud(font_path='c:/windows/fonts/malgun.ttf',
          background_color='white',
          stopwords=['analysis','big'],
          width=800, height=400).generate(text)

wordcloud.to_file('wordcloud2.png')

#{단어:빈도수} 사전
wordcloud.process_text(text)

#%% Counter 기반 (단어:빈도수) 사전 생성해서 word cloud
f = open('data/news.txt','r',encoding='utf-8')
corpus = f.readlines()
f.close()

corpus_morph=[]
for doc in corpus:
    lst=doc.split()    
    #글자수(음절)가 3이상이면 포함
    nou=[n for n in lst if len(n) >=3] #n이 분할된(split) 단어
    corpus_morph.append(nou)

corpus_morph

'''
[['현충원', '방명록'], ['서울시', '대한민국', '당선자', '코로나', '서울시']]
'''
#문자열
#Counter('abcdeabcdabcaba') : 문자마다 개수 집계

from collections import Counter
#단일리스트
#Counter('abcdeabcdabcaba') : 단어요소마다 개수 집계
counter =  Counter(['현충원', '방명록'])

#다중리스트
#단어들을 가지는 다중리스트를 Counter에 입력
#단어:빈도수의 사전을 가지는 Counter가 생성

counter =  Counter()
for m in corpus_morph:
    counter.update(m) #카운터에 추가
#elements() : 단어들
#sorted() : 정렬
print(sorted(counter.elements()))

#빈도수 상위 3개 
#(단어,빈도수) 튜플을 가지는 리스트 
counter.most_common(3)

#{단어:빈도수} 사전
#생성함수로 generate_from_frequencies() 사용
#stopwords 미반영,사전에 방명록 항목제거 
tf = dict(counter.most_common(3))
#%%
#del tf['현충원']
#tf.pop('방명록')
wordcloud = WordCloud(font_path='c:/windows/fonts/malgun.ttf',
          background_color='white',
          stopwords=['방명록']).generate_from_frequencies(tf)

import matplotlib.pyplot as plt
plt.figure(figsize=(22,22)) #이미지 사이즈
#wordcloud 이미지 출력
#이미지의 경계선 계단현상을 보간(interpolation)해서 부드럽게 표현
plt.imshow(wordcloud, interpolation='lanczos') 
plt.axis('off') #x, y 축 숫자 제거

#%% DataFrame word cloud
#generate_from_frequencies(사전)
#DataFrame -> 사전
df = pd.DataFrame({'keyword':['서울시','현충원','방명록'],
                                'count':[2,1,1]})


df = df.drop(2)
# df2 = df.copy()
# df2.drop(['count'], axis=1,inplace=True)

#결과 이상
#예를 들어 벨류값 ['서울시','현충원','방명록']도 사전구조로 변환
#tf = df.to_dict()
#단어사전 {'서울시': 2, '현충원': 1, '방명록': 1} 을 추출
tf = df.set_index('keyword').to_dict()['count']
wordcloud = WordCloud(font_path='c:/windows/fonts/malgun.ttf',
          background_color='white'
          ).generate_from_frequencies(tf)

import matplotlib.pyplot as plt
plt.figure(figsize=(22,22)) #이미지 사이즈
#wordcloud 이미지 출력
#이미지의 경계선 계단현상을 보간(interpolation)해서 부드럽게 표현
plt.imshow(wordcloud, interpolation='lanczos') 
plt.axis('off') #x, y 축 숫자 제거
plt.savefig('wc.png')

