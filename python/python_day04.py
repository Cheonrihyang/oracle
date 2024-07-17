# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:45:32 2024

@author: ORC
"""

#%% while

#홀수만 출력
a = 0;
while a<=10:
    if a%2==0:
        a+=1
        continue
    print(a)
    a+=1

#1~10합 산출
su=1
hap=0
while su<=10:
    hap+=su
    su+=1
print("1~10의 합",hap)

10*(1+10)/2

#%% for

#배열을 iterator방식으로 하나씩 꺼내온다
it = iter([1,2,3])
next(it)

#같은 방식을 사용한다
for i in [1,2,3]: #리스트
    print(i)
for i in {1,2,3}: #set
    print(i)
for i in (1,2,3): #튜플
    print(i)
for i in "123": #문자열
    print(i)
    
a=[(1,2),(3,4),(5,6)]
#기본은 튜플 괄호는 생략 가능
for [first,last] in a: #언패킹
    print(first + last)
    
#%% 

#1부터 11까지
a=range(1,11) 
a=range(11) #시작값은 생략가능(기본값 0)

#1부터 11까지 2씩 증가
a=range(1,11,2) #홀수
a=range(0,11,2) #짝수

#10에서 1까지 1씩 감소
a=range(10,0,-1)
a=list(a)

#리스트로 변환
a=list(a)

#range를 이용한 for문
add=0
for i in range(1,11):
    add+=i
print(add)
a = range(1,11) 
a = list(a)

for idx in range(len(a)):
    if a[idx] < 6:
        print(a[idx])

for i in a:
    if i < 6:
        continue
    print(i)
        
#%% 이중 for

#구구단
for i in range(1,10):
    for j in range(1,10):
        print((i*j), end=" ")
    print('')
    
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

#언패킹식
for ho1,ho2 in apart[::-1]:
    print(ho1,ho2)
    
#%% 딕서녀리 for문

date = ['09/05', '09/06', '09/07', '09/08', '09/09']
close_price = [10500, 10300, 10100, 10800, 11000]

#dict(zip())으로 딕셔너리 생성
mydict = dict(zip(date,
                  close_price))
for key in mydict.keys():
    print(key)
##반복변수의 기본은 키 
for key in mydict:
    print(key)

for value in mydict.values():
    print(value)
    
for key in mydict.keys():
    print(key,mydict[key])

#언패킹
for key,vlaue in mydict.items():
    print(key,vlaue)
    
#%% 조건부 표현식
score=50

#변수=참일때작동 if 조건 else 거짓일때작동 
message = "success" if score >= 60 else "failure"
print(message)

#%% 리스트 컴프리헨션
a = [1,2,3,4]

#리스트 = [표현식 for 변수 in 리스트 if 조건]
#결과 리스트 내에 연산식과 for포함
result = [num * 3 for num in a if num%2==0]
print(result)

a = [1,2,3,4]
#else가 있는 조건문을 사용할땐 조건부 표현식을 사용한다.
#리스트 = [조건부 표현식 for 변수 in 리스트]
result = [num * 3 if num%2==0 else num * 4 for num in a]

a = [1,2,3,4]

#%% 퀴즈

#정수형 여부
a=('1','2','3','4','A','B')
type('1')
'1'.isdigit()

#요소가 정수이고 짝수인 요소만 추출하여 정수형 정수 출력
a=(1,2,3,4,'A','B')
a=('1','2','3','4','A','B')
result = [int(i) for i in a if i.isdigit()==True and int(i)%2==0]

#%% 사전 내포
id_name = {1:'박진수',2:'강만진',3:'홍수정'}

#키,벨류 순서 바꾸기
{id_name[num]:num for num in id_name}
{val:num for num,val in id_name.items()}

