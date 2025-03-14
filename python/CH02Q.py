# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:59:48 2024

@author: ORC
"""

#%% 1

#홍길동씨의 과목별 점수는 다음과 같다. 홍길동 씨의 평균 점수를 구해 보자.
#과목 점수
#국어 80
#영어 75
#수학 55
score={'국어':80,'영어':75,'수학':55}
print((score['국어']+score['영어']+score['수학'])/3)

#%% 2

#자연수 13이 홀수인지, 짝수인지 판별할 수 있는 방법에 대해 말해 보자.
13%2
#출력 결과가0이 나오면 짝수 1이 나오면 홀수

#%% 3

#홍길동 씨의 주민등록번호는 881120-1068234이다. 
#홍길동 씨의 주민등록번호를 연월일(YYYYMMDD) 부분과 
#그 뒤의 숫자 부분으로 나누어 출력해 보자.
pin = "881120-1068234"
yyyymmdd = pin.split("-")[0]
num = pin.split("-")[1]
print(yyyymmdd)
print(num)

#%% 4

#주민등록번호 뒷자리의 맨 첫번째 숫자는 성별을 나타낸다.
#주민등록번호에서 성별을 나타내는 숫자를 출력해 보자.
pin = "881120-1068234"
print(pin[pin.index('-')+1])

#%% 5

#다음과 같은 문자열 a:b:c:d가 있다. 
#문자열의 replace 함수를 이용하여 a#b#c#d로 바꿔 출력해 보자
a="a:b:c:d"
b=a.replace(":", "#")
print(b)

#%% 6

#[1,3,5,4,2] 리스트를 [5,4,3,2,1]로 만들어 보자.
a=[1,3,5,4,2]
a.sort()
a.sort(reverse=True)
print(a)

#%% 7

#['Life','is','too','short'] 리스트를 Life is too short 문자열로 만들어 출력해 보자.
a=['Life','is','too','short']
a=" ".join(a)
print(a)

#%% 8

#(1,2,3)튜플 값 4를 추가하여 (1,2,3,4)를 만든 후 출력해 보자.
a=(1,2,3)
a=a+(4,)
print(a)

#%% 9

#다음과 같은 딕셔너리 a가 있다.
a = dict()
a
#다음 중 오류가 발생하는 경우를 고르고, 그 이유를 설명해 보자.
a['name']='python'
a[('a',)]='python'
a[[1]]='python'
a[250]='python'

a[[1]]='python'
#키는 가변형이면 안됨

#%% 10

#딕셔너리 a에서 'B'에 해당하는 값을 추출해 보자.
a={'A':90,'B':80,'C':70}
result = a.pop('B')
print(a)
print(result)

#%% 11

#a리스트에서 중복 숫자 제거
a=[1,1,1,2,2,3,3,3,4,4,5]
aSet = set(a)
b=list(aSet)
print(b)

#%% 12

#파이썬은 다음처럼 동일한 값에 여러 개의 변수를 선언할 수 있다.
#다음과 같이 a,b 변수를 선언한 후 a의 두번째 요솟값을 변경하면 b 값은 어떻게 될까?
#그리고 이런 결과가 나타나는 이유를 설명해 보자.

a=b=[1,2,3]
a[1]=4
print(b)

#얕은복사라 가르키는 주소값이 같아서 생기는 일이다.
#깊은복사를 사용하여 주소값이 다클론을 만들면 이런일이 생기지 않는다.