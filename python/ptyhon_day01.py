# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:55:58 2024

@author: ORC
"""

#%%
# =============================================================================
# 
# =============================================================================

# 결과값, 입력값
a = 3
b = 4
print("a+b는 :",a + b)
print("a-b는 :",a - b)
print("a*b는 :",a * b)
print("a/b는 :",a / b)
print("a**b는 :",a ** b)
print("a%b는 :",a % b)
print("a//b는 :",a // b)

#%% 사칙 연산 등등1
import math
반지름 = 6.0
원의넓이 = 반지름**2*math.pi
print("반지름이",반지름,"cm인 원의 넓이는",원의넓이,"cm")

#%% 사칙 연산 등등2
#변경내역 : 입력하면 생성
#정수문자열 -> int
import math
반지름 = float(input("반지름>>")) 
원의넓이 = int(반지름**2*math.pi)
print("반지름이",반지름,"cm인 원의 넓이는",원의넓이,"cm")

#%%
#원의 둘레 계산 프로그램 작성
import math
r = 6
둘레 = 2*r*math.pi
print("원의 둘레는",둘레)

#%%
#원의 부피 계산 프로그램 작성
import math
r = 6
h = float(input("높이입력"))
부피 = r**2*math.pi*h
print("부피는",부피)

#%%
#실수문자열 -> int
#type(int(input("반지름:")))
#소수이하 자리수 절삭
반지름 = int(float(input("반지름>>"))) 
print(반지름)

#%%
big = int(float('1.23'))
small = float('1.23') - big
small = round(small,2)
print(big,small)

#%%
a = 1
a+=1

#%% 문자열
# 여러 줄의 문자열을 변수에 대입
s = '''Life is too short,
 You need python'''
multiline = 'Life is too short,\n You need python'

#구분선
print("="*50)
#문자열 길이
len(s)

#%%
s = "CODE"
ln = len(s)
print("="*ln)
print(s)
print("="*ln)

#%%
a = 'Life is too short,\n You need python'
# 끝번호에 해당하는 문자는 포함하지 않음.
# a[시작번호:끝번호+1]
print(a[0:4])
print(a[19:])
print(a[:17])

a = "20230331Rainy"
year = a[0:4]
day = a[4:8]
weather = a[8:]

#%%
a='hobby'
print(a.count('bbb'))

a = "Python is the best choice"
print(a.find('b'))
print(a.find('k'))

print(a.index('b'))
print(a.index('k'))

#%% 퀴즈
#이메일 주소에서 아이디를 추출
email = input("이메일 입력")
print(email[0:email.find('@')])

#%%
",".join(['a' , 'b' , 'c' , 'd'])

#%%
a = "Life is too short"
a = a.split()
b = "a:b:c:d"
b = b.split(":")

#%%
MSG_FORMAT = "{}번 원반을 {}에서 {}로 이동"


def move(N, start, to):
    print(MSG_FORMAT.format(N, start, to))

def hanoi(N, start, to, via):#2 a c b
    if N == 1:
        move(1, start, to)
    else:
        hanoi(N-1, start, via, to) #1 a b c
        move(N, start, to)
        hanoi(N-1, via, to, start)

hanoi(5, 'A', 'C', 'B')

#%%
a='test'
#모두 대문자
a=a.upper()
#모두 소문자
a=a.lower()
a='                        sfasdasd??             '
#왼쪽에 있는 한 칸 이상의 연속된 공백과 '?' 모두 지운다
a=a.strip('? ')
a='        g      i      '.strip()
#\n도 제거해준다
a='           \nhi      \n'.strip()
#공백제거와 단어변환
a = "Life is too short"
a=a.replace(" ", "")
a=a.replace("Life","Test")

a="test.py"
#.py를 찾는다
a.find(".py")==-1
#.py로 끝나는가
a.endswith('.py')
#test로 시작하는가
a.startswith('test')

#이메일 주소에서 아이디와 도메인을 split을 이용해서각각 추출
email = input("이메일 입력")
email = email.split('@')
