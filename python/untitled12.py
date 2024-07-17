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
"""Life is too short, You need python"""
