# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:45:16 2024

@author: ORC
"""
#%% set 자료형

s1 = {1,2,3}
s1 = set([1,2,3])
s1.add(4)
#값 여러개 추가
s1.update([4,5,6])
s1.remove(2)
#l이 중복되어 1개만 적용된다.
s2=set("Hello")
#맨 앞의 값이 튀어나온다
s2.pop()

#인덱싱 접근하려면 리스트나 튜플로 변환해야 한다.
l=[1,2,3]
#리스트>튜플,set
s=set(l)
t=tuple(l)
#set>리스트
l=list(s)

#%% 퀴즈

#1. 2016년 11월 영화 예매 순위 기준 top3는 다음과 같다. 
#영화 제목을 movie_rank 이름의 튜플에 저장하라. (순위 정보는 저장하지 않는다.)
movie_rank=("닥터 스트레인지","스플릿","럭키")

#2. 다음 코드를 실행해보고 오류가 발생하는 원인을 설명하라.
t = (1, 2, 3)
t[0] = 'a'
#튜플은 수정이 불가능하다

#3. 변수 t에는 아래와 같은 값이 저장되어 있다. 변수 t가 ('A', 'b', 'c') 튜플을 가리키도록 수정 하라.
t = ('a', 'b', 'c')
t=list(t)
t[0]='A'
t=tuple(t)

#4. 다음 리스트를 튜플로 변경하라.
interest = ['삼성전자', 'LG전자', 'SK Hynix']
interest = tuple(interest)

#5. a리스트에서 중복된 숫자들을 제거후 내림정렬 리스트 출력
a = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5]
a=set(a)
a=list(a)
a.sort(reverse=True)

#%% 딕셔너리

dic = {'name': 'pey', 'phone': '010-9999-1234', 'birth': '1118'}
#딕셔너리 추가
dic[2]='b'
dic[3]=[1,2,3]

#딕셔너리 삭제
del dic[2]
del dic[3]

#딕셔너리 검색
dic['name']
dic.get('name1','무명') #(key,디폴트)
'name1' in dic #포함하는지 검색

#딕셔너리 수정
dic['name']='update'
dic['phone']='010-9999-1235'

#여러요소 수정,추가
dic.update({'phone': '010-9999-1236', 'birth': '1234', 'addr': '서울'})

#키값만 불러오기
dic.keys()
type(dic.keys()) #리스트가 아님
list(dic.keys()) #변환이 필요

#벨류값만 불러오기
dic.values()
type(dic.values())
list(dic.values())

#정렬
#dic.sort() #미지원
#키 순서대로 요소 정렬된 딕셔너리
dict(sorted(dic.items()))

#벨류 순서대로 요소 정렬된 딕셔너리
import operator
sorted(dic.items(),key=operator.itemgetter(1),reverse=True)

#key,values 쌍을 합친 딕셔너리
dict(zip(list(dic.keys()),list(dic.values())))

#%% 퀴즈

#1. 다음 아이스크림 이름과 희망 가격을 딕셔너리로 구성하라.
#이름	희망 가격
#메로나	1000
#폴라포	1200
#빵빠레	1800
ice={'메로나':1000,'폴라포':1200,'빵빠레':1800}

#2. 딕셔너리에 아래 아이스크림 가격정보를 추가하라.
#이름	희망 가격
#죠스바	1200
#월드콘	1500
ice['죠스바']=1200
ice['월드콘']=1500

#3. 딕셔너리를 사용하여 메로나 가격을 출력하라.
#실행 예:
#메로나 가격: 1000
print("메로나 가격:",ice['메로나'])

#4. 딕셔너리에서 메로나의 가격을 1300으로 수정한 후  메로나를 삭제하라.
ice['메로나']=1300
del ice['메로나']

#5. 딕셔너리에서 key 값으로만 구성된 리스트를 생성하라.
icelist=list(ice.keys())

#6. 딕셔너리의 아이템을 키값을 기준으로 정렬하라.
dict(sorted(ice.items()))

#%% 딕셔너리 배열

#벨류에 배열 넣기
ice['빵빠레']=[1800,1300]
ice['빵빠레'][1]

ice2={'메로나':1000,'폴라포':1200,'빵빠레':1800}
ice3={'메로나':1000,'폴라포':1200,'빵빠레':1800}

#딕셔너리 배열
icearr=[ice,ice2,ice3]
icearr[0]['빵빠레'][1]

#%% 패킹과 언패킹

#배열,튜플 요소를 변수에 나눈다
lst = [1,2,3]

#배열 언패킹
a,b,c = lst

#%% 퀴즈

#원주율수,오일러수 패킹과 언패킹
math = (3.14,2.718)
q,w=math

#%% bool

#값이 없으면 false
bool([])
bool([1])
bool('')
bool(1)
bool(2)
bool(3)
bool(0)#false

a=None
bool(a)

#%% 배열 복사

#얕은복사
a = [1,2,3]
b = a
id(a)
id(b)
b.insert(0, "test")

#깊은복사
from copy import copy
b=copy(a)
c=a[:]
id(a)
id(b)
id(c)

#%% if문
money = True

#중괄호 대신 들여쓰기
if money:#bool()디폴트
    print("택시를 타고 가라")
else:
    print("걸어 가라")

money = 2000
if money >= 3000:
    print("택시를 타고 가라")
else:
    print("걸어가라")

#or연산    
money = 2000
card = True
if money >= 3000 or card:
    print("택시를 타고 가라")
else:
    print("걸어가라")

#포함 여부 true false 반환
1 in [1,2,3]
1 not in [1,2,3]

#pass를 쓰면 아무일도 안일어나고 지나감
if money >= 3000 or card:
    pass
else:
    pass

#%% 다중조건 if

#elif
pocket = ['paper', 'cellphone']
if 'money' in pocket:
    print("택시를 타고 가라1")
else:
    if card:
        print("택시를 타고 가라2")
    else:
        print("걸어가라")
#동일하게 작동
pocket = ['paper', 'cellphone']
if 'money' in pocket:
    print("택시를 타고 가라1")
elif card:
    print("택시를 타고 가라2")
else:
    print("걸어가라")
    
#%% 퀴즈

#1. 투자 경고 종목 리스트가 있을 때 사용자로부터 종목명을 입력 받은 후 
#해당 종목이 투자 경고 종목이라면 '투자 경고 종목입니다'를 
#아니면 "투자 경고 종목이 아닙니다."를 출력하는 프로그램을 작성하라.
warn_investment_list=["Microsoft","Google","Naver","Kakao","SAMSUNG","LG"]
if input("종목명 입력") in warn_investment_list:
    print('투자 경고 종목입니다.')
else:
    print('투자 경고 종목이 아닙니다.')
    
#2. 사용자로부터 시간을 입력 받고 정각인지 판별하라.
if input("시간입력")[3:]=='00':
    print('정각 입니다.')
else:
    print('정각이 아닙니다.')