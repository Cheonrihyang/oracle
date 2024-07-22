# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:56:03 2024

@author: ORC
"""

#%% plot

import matplotlib.pyplot as plt

y=[20,30,40]
x=['kim','lee','kang']

#그래프생성
plt.plot(x, y)

#제목,x라벨,y라벨 설정
plt.title("Title")
plt.xlabel('X')
plt.ylabel('Y')

#%% 각종 설정

import matplotlib.pyplot as plt

y=[20,30,-40]
x=['안지영','홍지수','황예린']

#css설정가능
plt.title("세명 학생의 영어 성적",fontdict={
                                'size':18,
                                'color':'green',
                                'weight':'bold'
                                })
plt.xlabel('학생명')
plt.ylabel('점수')

#플롯 한글 깨짐방지 
plt.rc('font', family='Malgun Gothic') 
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
#글꼴크기
matplotlib.rcParams['font.size']=16
#음수부호 보이기(플롯 한글 깨짐방지 설정하면 음수부호 깨짐 )
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False



#그래프생성
plt.plot(x, y)

#%% 여러 개의 그래프를 범례(legend)와 함께 한 개의 창(플롯)으로 그리기

import matplotlib.pyplot as plt

x = ['안지영', '홍지수', '황예린']
y1 = [90, 85, 88] #국어
y2 = [85, 88, 90] #영어

#플롯(axes)의 배경색 노랑 지정 
plt.rcParams['axes.facecolor'] = 'y'
#플롯 한글 깨짐방지 
plt.rc('font', family='Malgun Gothic') 

# plt.plot(x,y1,c='red',label='국어') #범례label
# plt.plot(x,y2,c='green',label='영어')#범례label
# plt.legend()
plt.plot(x,y1,c='red') 
plt.plot(x,y2,c='green')
#plt.legend(['국어','영어'])
plt.legend(['국어','영어'],loc='upper center')#범례label과 위치
#fontdict : 폰트스타일 사전
plt.title('세명 학생의 국어,영어 성적',fontdict={'size':18,'color':'green'})
plt.xlabel('학생명')
plt.ylabel('성적')

#%% 여러 개의 그래프를 범례(legend)와 함께 한 개의 플롯으로 그리기 2

import matplotlib.pyplot as plt


#플롯 한글 깨짐방지 
plt.rc('font', family='Malgun Gothic') 
#플롯(axes)의 배경색 노랑 지정 
plt.rcParams['axes.facecolor'] = 'yellow'

x = ['안지영', '홍지수', '황예린']
y1 = [90, 85, 88] #국어
y2 = [85, 88, 90] #영어
y3 = [85, 97, 78] #수학
# plt.plot(x,y1,c='red',label='국어') #범례label
# plt.plot(x,y2,c='green',label='영어')#범례label
# plt.legend()

#linestyle=':'점선 ,-- 파선 - 실선
#marker=d:다이아몬드
#plt.plot(x,y1,c='#ff0000',linestyle='-', marker='o') 
#plt.plot(x,y1,c='red',linestyle='-', marker='o') 
#plt.plot(x,y2,c='green',linestyle=':', marker='x') 
# style 값들을 하나의 문자열로 지정(c=red  marker='s' linestyle='--') 
plt.plot(x,y1,'ro-')
plt.plot(x,y2,'gx:')

#plt.plot(x,y2,'gx--') #가능
#plt.plot(x,y2,'rs--') 
plt.plot(x,y3,c='magenta',linestyle=':',marker='s')
plt.legend(['국어','영어','수학'])#범례label과 위치
#fontdict : 폰트스타일 사전
plt.title('세명 학생의 세 과목 성적',fontdict={'size':18,'color':'green'})
plt.xlabel('학생명')
plt.ylabel('성적')
#축의 범위
plt.ylim(0,100) #현 예제는 눈금값 20
#축의 눈금값(tick) 지정
plt.yticks(range(0,100,10))#눈금값 10

#그리드 표시
plt.grid(True)

#%% bar

import matplotlib.pyplot as plt

y=[70,88,90]
x=['안지영','홍지수','황예린']
plt.rc('font', family='Malgun Gothic') 
plt.title("세명 학생의 영어 성적",fontdict={
                                'size':18,
                                'color':'green',
                                })

#수직방향
#plt.xlabel('학생명')
#plt.ylabel('성적')
#plt.bar(x, y,width=0.7,color='green') 

#수평방향
plt.ylabel('학생명')
plt.xlabel('성적')
plt.barh(x, y,height=0.7,color='green')

plt.grid(True)

#%% scatter

import matplotlib.pyplot as plt

y=[70,88,90]
x=['안지영','홍지수','황예린']
plt.rc('font', family='Malgun Gothic') 

plt.scatter(x,y)

x=[1,2,3,4]
y=[2*i for i in x]
plt.scatter(x,y)

#%% pie

plt.rc('font', family='Malgun Gothic') 
labels = ['개구리', '돼지', '개', '고양이'] #요소
sizes = [14.5, 30.5, 45, 10] # 요소값의 비율, 요소합이 100을 넘으면 자동 비율 계산
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
plt.title("반려동물 Pie Chart")

#startangle 90 :그래프의 시작 12시에서 반시계방향 요소 표시 개구리,돼지 이런순서
#autopct='%1.1f%%' : 요소값 소수점이하 1자리수 
explode=(0,0.1,0,0)
plt.pie(sizes,labels=labels, colors=colors,
        autopct='%1.1f%%',startangle=90,explode=explode) 







#%% 데이터프레임

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10, 4).cumsum(axis=0),
columns=["A", "B", "C", "D"],
index=np.arange(0, 100, 10))
df.plot()

s3 = pd.Series(np.random.normal(0, 1, size=200))
s3.hist()

x1 = np.random.normal(1, 1, size=(100, 1))
x2 = np.random.normal(-2, 4, size=(100, 1))
X = np.concatenate((x1, x2), axis=1)
df3 = pd.DataFrame(X, columns=["x1", "x2"])
#plt.scatter(df3["x1"], df3["x2"])
df3.plot.scatter('x1','x2')

#%%

import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Malgun Gothic') 
x=np.arange(1,10)
plt.hist(x,bins=5)


#%% 리스트를 DataFrame 변환후 plt 시각화

import matplotlib.pyplot as plt
import pandas as pd

x = ['안지영', '홍지수', '황예린']
y = [85, 88, 90]
df = pd.DataFrame(data=y, index=x, columns=["성적"])

#각각의 리스트를 열로 간주하여  열들을 사전으로 묶음
data =  {"학생이름": x, "성적": y}
#사전을 DataFrame 변환
df = pd.DataFrame(data)

#plot(x열,y열)
plt.plot(df["학생이름"], df["성적"])
#위치(x,y)에 str문자열 표시
plt.text(x=0,y=df["성적"].mean(),s='성적평균=%d'%df["성적"].mean())

plt.plot(df.index, df["성적"])
#x가 생략되면 index가 x축
plt.plot(df["성적"])

#fontdict : 폰트스타일 사전
plt.title('세명 학생의 영어 성적', 
          fontdict={'size':18,
                    'color':'green'})

plt.xlabel("학생 이름")

#파일저장
plt.plot(df["학생이름"], df["성적"])
plt.savefig('score.png')#위 코드와 함께 한꺼번에 실행

#%% 퀴즈
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('iris.csv',encoding='utf-8')

#1. iris 데이터셋의  sepal_width 컬럼을 라인플롯한다.
plt.plot(df.index, df['sepal_width'])

#2. iris 데이터셋의  sepal_width 컬럼을 산포플롯한다.
plt.scatter(df.index,df['sepal_width'])

#3. iris 데이터셋의  꽃 품종이름 별로 sepal_width 컬럼의 평균을 바플롯한다.
plt.bar(df.groupby('species').groups.keys(),df.groupby('species').mean()['sepal_width'])

#4. stock.png 참조하여 파이플롯한다.
plt.pie([50,10,10,10,20],
        labels=['삼성전자','현대차','포스코','네이버','카카오'],
        colors=['gold','yellowgreen','pink','lightskyblue','red'],
        explode=[0,0.1,0,0,0],
        autopct='%1.1f%%',startangle=90,shadow=True) 
plt.savefig('pie.png')

#%% 히트맵
#값이 클수록 밝음

import matplotlib.pyplot as plt
import numpy as np

y=np.random.randn(50).reshape(5,10)
plt.matshow(y)
plt.colorbar()

cmap=plt.get_cmap('bwr')
plt.matshow(y, cmap=cmap)
plt.colorbar()

#%% 누적비율 막대그래프

import matplotlib.pyplot as plt

data = {
    "도시": ["서울", "서울", "서울", "부산", "부산", "부산", "인천", "인천"],
    "연도": ["2015", "2010", "2005", "2015", "2010", "2005", "2015", "2010"],
    "인구": [9904312, 9631482, 9762546, 3448737, 3393191, 3512547, 2890451, 263203],
    "지역": ["수도권", "수도권", "수도권", "경상권", "경상권", "경상권", "수도권", "수도권"]
}
df = pd.DataFrame(data)
pt = df.pivot_table( index='도시', columns='연도', 
               values='인구',fill_value=9762540, #널값채움
               )

#누적비율 막대그래프
pt.plot.bar(stacked=True)

#집계
pt = df.pivot_table( index='도시', columns='연도', 
               values='인구',fill_value=9762540,
               aggfunc = [np.sum],
               margins_name="합계",
               margins=True)
pt.plot.bar(stacked=True)

#%% 면적그래프

df=pd.DataFrame({
    '판매자수':[3,2,3,9,10,6],
    '구매자수':[5,5,6,12,14,13]
    },index=pd.date_range(start='2018/01/01',end='2018/07/01', freq='M'))

df.plot(kind='area')

#%% 퀴즈

import pandas as pd

df = pd.read_csv('population_2020.csv',encoding='utf-8')
cols = ['2020년02월_총인구수', '2020년02월_세대수', '2020년02월_남자 인구수', '2020년02월_여자 인구수']

for x in cols:
    df[x]=df[x].apply(lambda x : int(x.replace(',','').replace(' ','')))

pt=df[cols].sum()

#1. 전체 총인구수,총세대수,남자총인구수,여자총인구수 출력한다. 
print(pt)
#2. 서울,부산, 대구, 인천, 광주, 대전지역의 총인구수와 세대수를 막대플롯한다. 
pt=df.loc[:5,['2020년02월_총인구수','2020년02월_세대수']]
pt.plot.bar()
df[]