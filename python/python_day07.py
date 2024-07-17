# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:15:06 2024

@author: ORC
"""

#%% 퀴즈
#1. 문자열 PER (Price to Earning Ratio) 값을 실수로 변환할 때 에러가 발생한다.
#예외처리를 통해 에러가 발생하는 PER은 0으로 출력하라.

per = ["10.31", "", "8.00"]

for i in per:
    try:
        print(float(i))
    except:
        print(0)
        
#2. 아래의 코드에 대해서 예외처리를 사용하고 try, except, else, finally에 적당한 코드를 
#작성해보자. else와 finally는 임의의 적당한 문구를 print하면 된다.

per = ["10.31", "", "8.00"]

for i in per:
    try:
        print(float(i))
    except:
        print(0)
    else:
        print("문제없")
    finally:
        print("출력")
        
#%% 클래스 선언과 호출

class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, num):
        self.result += num
        return self.result

#실체 생성    
cal1 = Calculator()
cal2 = Calculator()

#객체가 제공하는 메소드 호출
cal1.add(3)
cal2.add(3)

#%% 사칙연산 계산기

class FourCal:
    def setdata(self, first, second):
        self.first = first
        self.second = second
        
a=FourCal()
a.setdata(3, 4)
a.first=7

#%% 클래스 은닉화

class FourCal:
    def setdata(self, first, second):
        self.__first = first
        self.__second = second
        
a=FourCal()
a.setdata(4, 2)
#__를 붙이면 조회시 접근 불가
print(a.__first)
#변경이 아닌__first 변수 추가 생성
a.__first=5
a.__third=6
print(a._FourCal__first)

#%% 생성자,게터,세터

#은닉화 필수
class FourCal:
    #생성자
     def __init__(self, first, second):
         self.__first = first
         self.__second = second
         
     @property #게터
     def first(self):
         return self.__first
     
     @first.setter #세터(변수명.setter)
     def first(self,first):
         self.__first = first
         
     @property
     def second(self):
         return self.__second
     
     @second.setter
     def second(self,second):
         self.__second = second
     
     def add(self):
         result = self.__first + self.__second
         return result
     def mul(self):
         result = self.__first * self.__second
         return result
     def sub(self):
         result = self.__first - self.__second
         return result
     def div(self):
         result = self.__first / self.__second
         return result

     def __str__(self):
         return f'첫번째 수={self.__first},두번째 수={self.__second}'

a=FourCal(0,0)
#실제 변수에 직접 접근
a.first = 4 #내부적으로 세터 호출
print(a.first) #내부적으로 게터 호출

a.second = 7
print(a.second)

a.add()
a.sub()
a.mul()
a.div()
print(a)

#%% 클래스 메소드

#은닉화 필수
class FourCal:
    #생성자
     def __init__(self, first, second):
         self.__first = first
         self.__second = second
         
     @property #게터
     def first(self):
         return self.__first
     
     @first.setter #세터(변수명.setter)
     def first(self,first):
         self.__first = first
         
     @property
     def second(self):
         return self.__second
     
     @second.setter
     def second(self,second):
         self.__second = second
     
     def add(self):
         result = self.__first + self.__second
         return result
     def mul(self):
         result = self.__first * self.__second
         return result
     def sub(self):
         result = self.__first - self.__second
         return result
     def div(self):
         result = self.__first / self.__second
         return result

     def __str__(self):
         return f'첫번째 수={self.__first},두번째 수={self.__second}'
     
     #클래스메소드는 self변수 사용 불가
     #객체간의 공유 정보
     @classmethod
     def set_plus_button(cls,plus): #클래스가 전달
         #self.__first=0 불가
         cls.__plus_button = plus
     @classmethod
     def get_plus_button(cls): #클래스가 전달
         #self.__first=0 불가
         return cls.__plus_button
    
#클래스메소드는 객체 생성없이 클래스명으로 호출 가능         
FourCal.set_plus_button('+')
print(FourCal.get_plus_button())
a=FourCal(0,0)
#실제 변수에 직접 접근
a.first = 4 #내부적으로 세터 호출
print(a.first) #내부적으로 게터 호출

a.second = 7
print(a.second)

a.add()
a.sub()
a.mul()
a.div()
print(a)

#%% 퀴즈
#1. 다음과 같이 코드가 동작하도록 사람 클래스를 수정하라.
#class 사람:    pass

#human = 사람('김철수', '010-1234-5678')
#human.call()    
#김철수에게 010-1234-5678 발신

#human2 = 사람()
#human2.call()
#홍길동에게 010-1234-1234 발신

class 사람 :
    def __init__(self,name="홍길동",tel="010-1234-1234"):
        self.name=name
        self.tel=tel
        
    def call(self):
        print(f'{self.name}에게 {self.tel} 발신')
    
    def 개인정보(self):
        print(self.name[0],end="")
        for a in range(len(self.name[1:-1])):
            print("*",end="")
        print(self.name[-1])
human = 사람('김철수', '010-1234-5678')
human.call()

human2 = 사람()
human2.call()

#2. 다음과 코드가 동작하도록 사람 클래스를 수정하라.
#단 첫글자와 마지막 글자외에는 '*' 로 대신하여 개인정보를 블라인드 처리한다.
#human.개인정보("유종훈")
#유*훈
#세터 추가 필요
human.개인정보()
#유**라

#%% 모듈

#import 파일명 as 별명
import mod1 as mo
mo.add(1, 2)

#from 파일명 import 메소드
#from mod1 import add,sub
from mod1 import *
add(1, 2)
sub(1, 2)

#%% 패키지

#import game.echo
#game.echo.echo_test()

#from 패키지명 import 파일명 as 별명
from game import echo as ech
ech.echo_test()

#%% 외부 모듈 크롤링

import urllib.request as req
import bs4
html = req.urlopen('http://finance.naver.com/')    
bs=bs4.BeautifulSoup(html, 'html.parser')
#창제목 태그명 검색 
print(bs.find('title'))

#%% time 라이브러리

import time
start = time.time()
#실행시간 구하기
for i in range(10000):
    print(i)
    
#작동시간 현재시간 - 시작시간
print(time.time()-start)

#현재시간 구하기
print(time.localtime(time.time()))
time.ctime() #간단함

#출력포멧 직접 설정
time.strftime("%Y-%m-%d %A %H:%M:%S",time.localtime())

#%% datetime

from datetime import datetime
today = datetime.now()
#현재시간,포멧문자열
datetime.strftime(today,"%Y%m%d %A %H:%M:%S")

dt = datetime(2024,7,8,12,00,00)
datetime.strftime(dt, "%Y-%m-%d %A %H:%M:%S")

dt2 = datetime.strptime('2024-07-09 Monday 12:00:00',"%Y-%m-%d %A %H:%M:%S")

#시간 계산
import datetime
day1 = datetime.date(2024, 12, 14)
day2 = datetime.date(2024, 4, 5)
print(day1-day2)

#timedelta
td = datetime.timedelta(days=-3) #3일 이전
print(td)

td2 = datetime.timedelta(hours=3) #3시간 이후
print(td2)

#시간 계산2
today + td + td2

#%% 랜덤

import random
#0.0~1.0미만 실수 랜덤
random.random()

#1.0~20.5미만 실수 랜덤
random.uniform(1.0, 20.5)

#1~55 정수 랜덤 끝수 포함
random.randint(1, 55)

#배열 랜덤
random.choice(['a','b','c'])

data = [1,2,3,4,5]
random.sample(data, 2)
#섞기
random.sample(data, len(data))
random.shuffle(data)

#%% 퀴즈

#로또 시스템
lotto = list(range(1,46))
random.sample(lotto, 6)
