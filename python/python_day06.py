# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:12:44 2024

@author: 82103
"""

#%% 파일 io - write

f = open("새파일.txt", 'w')

for i in range(1, 11):
    data = "%d번째 줄입니다.\n" % i
    f.write(data)   # 인자가 문자열만 가능
    f.flush()   # 강제저장
f.writelines('11번째 줄입니다. \n')
# 텍스트 문자열을 요소로 가지는 리스트가 인자로 올 수 있다
f.writelines(['12번째 줄\n', '13번째 줄\n'])
# 저장 후 닫기
f.close()



#%% 파일 io - read

f = open("새파일.txt", 'r')
line = f.readline()
print(line)
f.close()

# 파일의 모든 라인을 읽어서 각 라인을 요소로 갖는 리스트로 리턴
f = open("새파일.txt", 'r')
lines = f.readlines()[:4]
for line in lines:
    print(line.strip('\n'))
f.close()

# 파일의 내용 전체를 문자열로 리턴
f = open("새파일.txt", 'r')
data = f.read()
print(data)
f.close()


#%% 읽고 쓰기 기능 w+(파일 미존재이면 생성) r+(파일 없으면 오류)

f = open("새파일.txt", 'w+', encoding=('utf-8'))

f.write('1번째 줄')    # 인자가 문자열
f.flush()               # 강제저장
print("현재 커서 위치: ", f.tell())
f.seek(0)   # 커서 위치 (0번) 으로 이동
data = f.read()
print(data)
f.close()


#%% 절대경로

path = "C:\\Users\\82103\\.spyder-py3\\새파일2.txt"
path = "C:/Users/82103/.spyder-py3/새파일3.txt"
f = open(path, 'w+', encoding=('utf-8'))
f.close()


#%% with 블록을 벗어나는 순간 열린 파일 객체 f가 자동으로 close

with open("foo.txt", "w") as f:
    f.write("Life is too short, you need python")


#%% 퀴즈

#1. '매수종목1.txt' 파일을 생성한 후 다음과 같이 종목코드를 파일에 써보자.
#005930
#005380
#035420
strs = ["005930","005380","035420"]
f = open("test.txt",'w', encoding=('utf-8'))
for tt in strs:
    f.write(tt + '\n')
f.close()


#2. 생성한 '매수종목1.txt' 파일을 읽은 후 종목코드를 리스트에 저장해보자.
f = open("test.txt", 'r', encoding=('utf-8'))
read_data = list(f.read().splitlines())
print(read_data)
f.close()




# 1번
stocks = ["005930", "005380", "035420"]
f = open("매수종목1.txt", mode='w', encoding=('utf-8'))
for stock in stocks:
    f.write(stock + '\n')
f.close()





# 2번
f = open("매수종목1.txt", 'r', encoding=('utf-8'))
read_data = list(f.read().splitlines())
print(read_data)
f.close()

# 강사님 답안
f = open("매수종목1.txt", 'r', encoding=('utf-8'))
lines = f.readlines()
codes = [line.strip() for line in lines]
print(codes)
f.close()

# 3번
dic_stocks = {"005930":"삼성전자", "005380":"현대차", "035420":"NAVER"}
with open("매수종목2.txt", 'w') as f:
    for stock, name in dic_stocks.items():
        f.write(f'{stock} {name}\n')


# 4번
read_dic = {}
with open("매수종목2.txt", 'r', encoding=('utf-8')) as f:
    lines = f.read().splitlines()
    for line in lines:
        stock, name = line.split()
        read_dic[stock] = name
print(read_dic)


# 5번
# test_input.txt 파일에 데이터 생성
with open("test_input.txt", 'w', encoding=('utf-8')) as f:
    f.write("가나다\n")
    f.write("라마바\n")

# test_input.txt 파일에 있는 데이터 읽어오기
with open("test_input.txt", "r", encoding=('utf-8')) as f:
    lines = f.readlines()
# 읽은 값을 역순으로 정렬
reversed_lines = reversed(lines)

# 역순으로 정렬한 데이터를 test_output.txt에 저장
with open("test_output.txt", "w", encoding=('utf-8')) as f:
    for line in reversed_lines:
        f.write(line)




#%% 예외

try:
    4 / 0
except ZeroDivisionError as e:
    print(e) #파이썬 오류 메세지
    print("(정수는 0으로 나누면 안됩니다.)")
print("종료")


#%% 여러개의 오류 처리하기
try:
    a = [1,2]
    print(a[1])
    4/0
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다.")
except IndexError:
    print("인덱싱 할 수 없습니다.")


#%% try ~else

try:
    age=int(input('나이를 입력하세요: '))
except:
    print('입력이 정확하지 않습니다.')
else:
    if age <= 18:
        print('미성년자는 출입금지입니다.')
    else:
        print('환영합니다.')


#%% 보안성의 향상

# 3번 재입력 기회
# 3번째는 보안문자를 요구
# 4번쨰 잘못 입력 시 프로그램 강제 종료
# 단, 올바른 값(정수 문자열) 입력 시 반복 종료
import sys
isHuman = True #입력자 사람여부
sc='1234'#보안문자
for i in range(4):
    try:
        if isHuman:
            a = int(input("정수:"))
            print(a)
        else:
            a = int(input("정수:"))
        #정수 입력하더라도 
        #잘못된 보안문자 입력시 프로그램 종료
            b = input("보안문자:")
            if(b != sc):
                print('너 로봇')
                sys.exit()#프로그램 강제 종료  
            
    except ValueError as e:
        print(type(e))
        print(e)
        if(i==2):#세번잘못입력
            isHuman = False
            print("보안문자",sc)
        if(i==3):#마지막 네번잘못입력
            print('너 로봇')
            sys.exit()#프로그램 종료   
        continue;
    else:#예외미발생시(정수 입력)
        break
    
print('Good')