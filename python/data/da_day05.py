# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:52:15 2024

@author: ORC
"""

#%% Series

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

