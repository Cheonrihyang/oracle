# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:42:57 2024

@author: ORC
"""

#1. % 문자열포메팅
#화면 = 템플릿 % 콘텐츠
#정수 포메팅
print("I eat %d apples."%3)

#문자 포메팅
print("I eat %s apples."%'five')

import math
#실수 포메팅
print("I eat %f apples."%math.pi)
#실수 포메팅 자르기
print("I eat %.2f apples."%math.pi)
print("I eat %0.2f apples."%math.pi)

#다중 포메팅
print("I eat %f apples. %d "%(math.pi, 123))

#%% 포메팅

#포메팅 함수
#화면 = 템플릿.format(콘텐츠)
#%형식 문자 -> {콘텐츠 인덱스}
print("I eat {0} apples.".format(3))

number = 10
day = "three"
#다중 포메팅 함수
print("I ate {0} apples. so I was sick for {1} days.".format(number, day))
print("I ate {1} apples. so I was sick for {0} days.".format(number, day))

#실수 포메팅함수 슬라이싱
print("I eat {:0.2f} apples.".format(math.pi))

#f문자열 포메팅
name='홍길동'
age=30
print(f"내이름은 {name} 나이는 {age+10}살")
print(f"I ate {math.pi:0.4f} apples")

#%% 배열

#배열
d = [1,2,'life','is']
d[0]
a=[1,2,3]
a[0]+a[2]

#이중배열
a = [1,2,3,['a','b','c']]
a[3][1]

#리스트 슬라이싱
a = [1,2,3,4,5]
a = a[:2]
a = [1,2,3,4,5]
a = a[2:-1]
a = [1,2,3,4,5]
a = a[:]

#확장 슬라이싱
a[::2]
#역정렬
a[::-1] #원본유지
a.reverse() #원본변경

#배열 연산
a = [1,2,3]
b = [4,5,6]
a+b
a*3

#리스트 요소 추가
a = [1,2,3]
a.append([4,5]) #리스트 자체가 추가
a.extend([4,5]) #리스트 내용이 추가

#리스트 삽입
a.insert(0, 0)

#리스트 요소 끄집어 내기 pop
#맨 마지막 요소를 리턴하고 그 요소는 삭제
a = [1,2,3]
a.pop()

#remove(x)는 리스트에서 첫번째로 나오는 x를 삭제
a = [1,2,3,1,2,3]
a.remove(3)
#리스트 삭제 내장 함수
del a[0:2]
a[0:2]=[]

#수정
a=[1,2,3,1,2,3]
a[0:2]=[9,8]

#검색
a=[1,2,3]
a.index(3)

1 in a #검색후 true false 반
a.count(0) #0의 개수 없어도 됨
a.index(0) #없으면 에러남

#리스트 길이 함수
len(a)

#리스트 삭제
a=[]
a.clear()
del a[:]

#%%
#sort 함수는 리스트의 요소를 순서대로 정렬
a = [1,4,3,2]
#asc
a.sort()
#desc
a.sort(reverse=True)

#정렬된 새로운 리스트를 반환
sorted(a)
sorted(a,reverse=True)

#1. price 변수에는 날짜와 종가 정보가 저장돼 있다. 
#날짜 정보를 제외하고 가격 정보만을 출력하라. (힌트 : 슬라이싱)
price = ['20180728', 100, 130, 140, 150, 160, 170]
price[1:]
#2. 사용자가 단어를 콤마(,)로 구분해 입력하면 
#그 단어들을 문자표에 따른 사전순(알파벳순)으로 정렬해 
#하나의 문자열로 출력하는 프로그램을 작성한다.
fr='바나나,오렌지,사과'
fr=fr.split(",")
fr.sort()
','.join(fr)
#3. 오늘이 월요일이라고 하자. 오늘부터 10일 후에는 어떤 요일이 되는가?
# 나머지 연산자 %를 적극적으로 사용해보자. 
#요일리스트 생성후 나머지 연산자 %를 사용해보자. 
day=['일','월','화','수','목','금','토']
day[day.index('월')+10%7]

#%%
day=['일','월','화','수','목','금','토']
day_str = ' '.join(day)
day_str = ','.join(day)
#str -> list
day_lst=list(day_str)#문자리스트
day_lst=day_str.split(',')#단어리스트

#%%
str(True)

#%% 튜플
#요소값을 변경 불가
t1 = ()
t2 = (1,) #숫자가 하나일시 쉼표 필수
t3 = (1, 2, 3)
t4 = 1, 2, 3 #괄호 생략 가능
t5 = ('a', 'b', ('ab', 'cd'))
#t5[0]='f' #변경 불가
#del t5[0] #삭제 불가

#편법 수정,삭제
t3_list = list(t3)
t3_list[0]=4
t3_2 = tuple(t3_list)

#인덱싱
t3[1]
t3[1:]

#튜플연산 추가 확장은 가능
t3+(3,4)
t3*2
