# -*- coding: utf-8 -*-
#%% 함수

#함수선언 리턴타입 기재 안함
def add(a,b):
    return a+b

#함수호출
res = add(1,2)
print(res)

def add(a,b):
    return("%d, %d의 합은 %d입니다." % (a, b, a+b))
    
#이름이 같으면 최근에 호출된 함수가 호출됨
add(1,2)

#%% 매개변수 초기값 설정

def say_myself(name, age, man=True): 
    print("나의 이름은 %s 입니다." % name) 
    print("나이는 %d살입니다." % age) 
    if man: 
        print("남자입니다.")
    else: 
        print("여자입니다.")
    return None; #없으면 자동 생성
        
print(say_myself("홍길동", 30,False))

#기본값 매개변수는 뒤에 있어야함.
def say_myself2(name,man=True, age): 
    print("나의 이름은 %s 입니다." % name) 
    print("나이는 %d살입니다." % age) 
    if man: 
        print("남자입니다.")
    else: 
        print("여자입니다.")
        
#%% 메인함수와 서브함수

#리스트 평균 상회 리스트 출력
def getUpperList(num_list):
    print(num_list)
    hap = 0
    avg = 0.0
    for num in num_list:
        hap+=num
    print("합",hap)
    avg=hap/len(num_list)
    print("평균",avg)
    result = [num for num in num_list if num > avg]
    return result

def main():
    number_list = [10,20,30,40,50]
    res = getUpperList(number_list)
    print(res)
    
main()

#%% 가변 매개변수 함수

#매개변수에 *을 붙인다
def add_many(*args):
    print(args)#튜플로 패킹
    result = 0 
    for i in args: 
        result = result + i   # *args에 입력받은 모든 값을 더한다.
    return result 

add_many(1,2,3)
add_many(1,2,3,4,5,6,7,8,9,10)
#컬렉션 전달시 *을 앞에 붙여서 언패킹
add_many(*[1,2,3,4,5,6,7,8,9,10])

#키워드 매개변수

#*을 2개 붙여서 매개변수를 딕셔너리로 받음
def print_kwargs(**kwargs):
    print(type(kwargs))
    print(kwargs)

#키워드(문자여야함)와 같은 벨류의 나열
print_kwargs(a=1)
print_kwargs(name='foo',age=3)

member = {'name':'foo','age':3}
#딕셔너리 전달시 **을 앞에 붙여서 언패킹
print_kwargs(**member)

#%% 매개변수 혼합

#여러 타입의 매개변수를 동시에 사용
def print_args_kwargs(a,*args,**kwargs):
    print(a)
    print(args)
    print(kwargs)

print_args_kwargs(1,*[2,3],**{'four':4})


def print_args_kwargs(*args,a):
    print(a)
    print(args)
    
#%% 퀴즈

#입력 사전 개수에 상관없이 사전이 가지는 벨류값들의 평균 리턴 함수
#입력 사전들 
#{'one':1,'two':2}
#{'one':1,'two':2,'three':3}

def avg(**dic):
    num=0
    nlist=list(num for num in dic.values())
    for n in nlist:
        num+=n
    return num/len(dic)
    
avg(**{'one':1,'two':2,'three':3})

#%% 하나의 리턴값

#튜플 하나만 반환해줌
def add_and_mul(a,b):
    return a+b,a*b

add_and_mul(4,5)

#%% 전역변수와 지역변수

#a에 영향을 주지 못한다.
a = 0;
def vartest(a):
    a=a+1
    
vartest(a)
print(a)

#1 리턴을 이용해서 변경한다
def vartest(a):
    a=a+1
    return a

a=vartest(a)
print(a)

#2 global 이용(권장하지 않음)
a = 0;
def vartest():
    global a
    a=a+1
    
vartest()
print(a)
        
#%% 람다함수

#자가호출
(lambda a,b:a+b)(3,4)

#함수참조변수
x = lambda a,b:a+b
x(2,3)

#매개변수 없는 자가호출
(lambda :3+4)()

#%% 퀴즈
import math
r=2
(lambda x:x**2*math.pi)(r)
