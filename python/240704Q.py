# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:01:27 2024

@author: ORC
"""

#%% 퀴즈

#1. 리스트에 동물 이름 저장돼 있다.
list = ['dog', 'cat', 'parrot']
#for문을 사용해서 동물 이름의 첫 글자만 출력하라.
for ani in list:
    print(ani[0])

#2. 리스트에서 20 보다 작은 3의 배수를 출력하라 
list = [13, 21, 12, 14, 30, 18]
for a in list:
    if a<20 and a%3==0:
        print(a)

#3. 생년월일(19800123)을 입력받아서 해당년도가 윤년인지 판단하는 코드를 작성한다.
birth = input("생년월일 입력")
year = int(birth[:4])
#4의배수면 윤년 100의 배수면 평년이지만 400의배수면 윤년
if year%4==0 and year%100!=0 or year%400==0:
    print("윤년임")

#4. 월드컵은 4년에 한 번 개최된다. 
#range()를 사용하여 2002~2030년까지 중 월드컵이 개최되는 연도를 출력하라.
for year in range(2002,2030+1):
    if (year-2)%4==0:
        print(year)
        
#5. 리스트에 저장된 데이터를 아래와 같이 출력하라.
#301 호
#302 호
#201 호
#202 호
#101 호
#102 호
apart=[101,102],[201,202],[301,302]
for dong in reversed(apart):
    for ho in dong:
        print(ho,"호")