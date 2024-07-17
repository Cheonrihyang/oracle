# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 09:41:38 2024

@author: ORC
"""

#%% math

import math
math.pi
math.e

math.ceil(math.pi)
math.factorial(5)
math.floor(3.55)
math.trunc(3.55)
round(3.5)

#자연로그(윗수,밑수)
math.log(2,2)
#상용로그
math.log10(10)
math.log(math.e)

#e의 n제곱
math.exp(2)
math.pow(2, 3)

#루트
math.sqrt(2)

#삼각함수
math.sin(math.radians(90))
math.cos(math.radians(90))
math.tan(math.radians(90))
math.tanh(math.radians(90))

#라디안값 각도 반환
math.degrees(1)

#E가 발생할 확률을 P(E)
#E에 대한 정보앤트로피 I(E)
math.log2( P(E))

#확률이 높을수록 정보엔트로피가 낮다
math.log2(1)


#%% 실수값 계산

4.1-1.2 #값이 정확하지 않음

import decimal
import math
decimal.Decimal('4.1')-decimal.Decimal('1.2')

#실수 비교 연산
(4.1-1.2)==2.9 #false
decimal.Decimal('4.1')-decimal.Decimal('1.2')==decimal.Decimal('2.9') #true
math.isclose((4.1-1.2), 2.9) #ture

#%% 폴더/파일 os

import os
os.environ['PATH']

#존재여부 확인
os.mkdir('c:/pytest')
os.path.exists('c:/pytest')

f=open('c:/pytest/test.txt','w',encoding=('utf-8'))
f.close
f=open('c:/pytest/test2.txt','w',encoding=('utf-8'))
f.close

#파일 존재여부 확인
os.path.isfile('c:/pytest/test.txt')

#디렉토리 리스트 출력
os.listdir('c:/pytest')

#절대경로
os.path.abspath('ptyhos_day07.py')

#크기출력
path = 'c:/pytest/test2.txt'
os.path.getsize(path)

#생성일
import time
time.ctime(os.path.getatime(path))

#이름변경
os.rename('c:/pytest/test.txt', 'c:/pytest/testnew.txt')

#파일삭제
os.remove('c:/pytest/testnew.txt')
os.remove('c:/pytest/test2.txt')

#폴더삭제
os.rmdir('c:/pytest')

#%% 퀴즈

#경로를 입력받은 후 해당 경로에 존재하는 파일과 디렉터리 리스트를 구한다.
#단 디렉터리이면 뒤에 (d)를 붙이고 파일이면 뒤에 (f)를 붙인다.
import os
path=input("경로입력")
arr = os.listdir(path)
result = [arg+"d" if os.path.isdir(path+arg) else arg+"f" for arg in arr]

#%% 폴더/파일 shutil
import shutil

#백업파일 생성
shutil.copy('test.txt', 'test.txt.bak')
shutil.copytree('game','gamebak')

#폴더/파일 이동
shutil.move('test.txt.bak', 'c:/cha')
shutil.move('gamebak', './game/gamebak')

#해당경로 모두 삭제
shutil.rmtree('./game')

#%% 트레이스백

import traceback

#오류정보를 상세하게 보여줌
def a():
    return 1/0

def b():
    a()

def main():
    try:
        b()
    except:
        print("오류가 발생했습니다.")
        print(traceback.format_exc())

main()

#%% 지로브

import glob

#.py가 붙은 파일 전부 검색
glob.glob('*.py')

#하위디렉토리까지 다 검색
glob.glob('**/*.py',recursive=True)

#%% 피클

import pickle

#파일에 딕셔너리 저장
f = open("test.dat", 'wb')
data = {1: 'python', 2: '파이썬'}
data2 = {1: 'python2', 2: '파이썬2'}
pickle.dump([data,data2], f)
f.close()

#읽기
f = open("test.dat", 'rb')
data = pickle.load(f)
f.close()
print(data)

#%% 퀴즈

#2. 현재폴더내의 모든 .py파일을  sorted 함수로 정렬하여 .py파일의 절대경로를 출력한다.
#단 파일 크기순으로 정렬한다.(sorted 함수의 key 인자에 파일 크기 리턴함수를 적용)


import glob
import os
import operator

arr = glob.glob('*.py')
dic = {os.path.abspath(x):os.path.getsize(x) for x in arr}

dic=dict(sorted(dic.items(),key=operator.itemgetter(1)))

for i in dic:
    print(os.path.abspath(i)+" : "+str(dic[i]))
