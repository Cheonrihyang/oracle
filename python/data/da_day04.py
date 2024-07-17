# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:33:31 2024

@author: 82103
"""

# %% numpy (Numerical Python)
# 배열의 모든 원소가 같은 자료형
# 반복문을 작성할 필요 없이 전체 데이터 배열에 대해 빠른 연산
import numpy as np
a = [1, 2, 3, 4]

# 리스트 -> 배열
arr = np.array(a)

# 배열 -> 리스트
b = arr.tolist()


print(arr)
# array 원소만큼 자동으로 순환 계산
arr+1  # 모든 요소에 1을 더해버림

# %% numpy
a = [1, 2, 3, 4]
# 리스트 -> 배열
arr = np.array(a)
arr.ndim  # 1 - 몇차원 배열인지를 알려줌
arr.shape  # (4, ) - 행, 열 크기
arr.size  # 총 갯수
arr.dtype  # 배열의 데이터 타입 리턴

a = [
    [1, 2, 3],
    [4, 5, 6]
]
arr = np.array(a)
arr.ndim  # 1 - 몇차원 배열인지를 알려줌
arr.shape  # (4, ) - 행, 열 크기
arr.size  # 총 갯수
arr.dtype  # 배열의 데이터 타입 리턴

len(arr)  # arr.shape의 첫번째 요소

# %% dtype
# 값의 범위를 감안한 자료형
# int(파이썬 기본 자료형), int32, np.int, np.int32 모두 동일
# float(파이썬 기본 자료형), float64, np.float, np.float64 모두 동일

# int32 -> int64 자동 업캐스팅
a = [1, 2, 3, 40000000000]
arr = np.array(a)

a = [1, 2, 3, 4]
arr = np.array(a)
# 이미 만들어진 상태에서 추가할 경우 overflow 에러 발생
# OverflowError: Python int too large to convert to C long
np.insert(arr, 3, 40000000000)

# 숫자 크기에 따른 자료형 변환 (명시적인 변환)
arr2 = arr.astype(np.int64)
np.insert(arr2, 2, 40000000000)


# 실수형
a = [1, 2, 3.0, 4]
arr = np.array(a, dtype=np.float64)
arr = np.array(a, dtype=float)
arr2 = arr.astype(np.int32)  # int32 로 변환

# 문자형
a = [1, 2, 3.0, 4]
arr = np.array(a, dtype=np.str_)
arr = np.array(a, dtype=str)


# %% 인덱싱, 슬라이싱 - 1차 배열
# 1차 배열
arr = np.array(range(20))
# 인덱스
arr[0]
arr[-1]
# 연속적인 인덱스 범위
arr[0:4]


# 확장 슬라이싱
# [::인덱스증감치(step, 간격)]
arr[0:9:2]

# 비연속적인 인덱스 목록
arr[[0, 3, 7]]

# p.200 확인문제 - [2,3,4] 추출
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(arr[1:4])

# %% 인덱싱, 슬라이싱 - 2차 배열
# 2차 배열
# [행에 대한 인덱스, 열에 대한 인덱스]
a = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

arr = np.array(a)
arr[0:2, 0:2]

# 행기준
arr[0]
arr[0:2]
arr[2, ]
arr[2, :]
arr[-1, :]

# 열기준
arr[:, 2]
arr[:, -1]
arr[:, 0:2]

# 마지막 행 미포함
arr[:-1]

# 모든 행렬 조회
arr[:]
arr[0:3, 0:3]
arr[0:4, 0:5]  # numpy에서는 행과열을 초과해서 조회도 가능하다.

# 확장 연속적인 슬라이스
# [::인덱스증감치(step, 간격)]
a[0:3:2]
arr[0:3:2, 0:3:2]
arr[1:3:, ::2]

# 비연속적인 슬라이스
# [[행 인덱스 리스트], [열 인덱스 리스트]]
arr[[0, 1], [0, 1]]
arr[[0, 2], [1, 2]]

# 인덱스 목록과 인덱스 범위 조합
arr[[0, 2], 1:3]  # 0번행의 1:3, 2번행의 1:3


# %% Boolean indexing

lst = [
      [1,	2, 3],
      [4,	5, 6],
      [7,	8, 9]
]
a = np.array(lst)

# 배열 a에 대해 짝수면 True, 홀수면 False
bool_indexing = (a % 2 == 0)
print(bool_indexing)

print(a[bool_indexing])  # a 배열에 bool_indexing 적용

n = a[a % 2 == 0]  # bool_indexing a 배열에 바로 적용해서 출력
print(n)

a[a > 3]
a[a < 6]
a[(a > 3) & (a < 6)]  # and 연산자
a[(a == 3) | (a == 6)]  # or 연산자

# 선택할 행 인덱스 논리값
a[a[:, 2] > 3] 

arr[(arr[:,2] >=3) & (arr[:,2]<=6), :]
arr[(arr[:,2] >=3) & (arr[:,2]<=6), 0:2]


#%% 조건에 만족하는 요소의 인덱스배열 리턴
arr = np.array(range(20))
np.where(arr > 3)

lst = [
       [1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]
       ]
arr = np.array(lst)
np.where(arr == 9)
# 요소들의 (행인덱스들, 열인덱스들)
np.where(arr > 3)


#%% shape 변환
# 1차 배열을 2차 배열로 변환
arr = np.array(range(20))
arr.reshape((4,5)) # 새 2차배열 리턴 (원본값 안변함)
arr.resize((4,5)) # 호출한 배열을 2차배열로 변환 (원본값 변함)

arr.np.array(range(20))
arr.reshape((4,3)) # 사이즈가 맞지 않으면 변환 불가
arr.reshape(-1,5) # -1로 할경우 알아서 정해짐
arr.reshape(4, -1)

# 2차 배열을 1차 배열로 변환
lst = [
       [1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]
       ]
arr = np.array(lst)
arr.reshape(arr.size) # 새로운 1차배열 리턴
arr.reshape(-1,)
arr.flatten()
arr.resize(arr.size) # 인플레이스 함수 (원본 값이 바뀜)


#%% 배열생성
import numpy as np

# array + range
# np.array(range(20))
np.arange(20)
# 0으로 초기화
np.zeros((2,2))
# 1로 초기화
np.ones((2,2))
# full (배열의크기, 초기화 할 숫자)
np.full((2,2), 5)

# 단위행렬
# 대각선으로는 1이고 나머지는 0인 2차원 배열
np.eye(4)
np.identity(3)

#%% 등간격 배열
# 시작값과 끝값을 지정한 간격 수로 나눈 값들을 생성
np.linspace(0, 1, 5) #(시작, 끝, 간격 수)

# 비균등분포 난수 배열
# 6개 정수난수(0~9범위) 1차 배열
np.random.randint(0, 10, size=6)
# 6개 실수난수(0~1) 1차 배열
np.random.rand(6)

# 6개 정수난수(0~9범위) 2차 배열 (2x3)
np.random.randint(0, 10, size=(2,3))
# 6개 실수난수(0~1) 2차 배열 (2x3)
np.random.rand(2,3)


# 정규 분포를 따르는 난수 배열
# np.random.normal(정규분포평균, 표준편차, (행, 열) or 개수)
np.random.normal(0, 1, 6)
np.random.normal(0, 1, (2,3))

# 평균이 0이고, 표준편차가 1인 정규분포율 표준정규분포
np.random.randn(6)
np.random.randn(2,3)



#%% 1차 배열 CUD
lst1 = [1, 2, 3]
lst2 = [4, 5, 7]

arr1 = np.array(lst1)
arr2 = np.array(lst2)

arr = np.arange(1,5)
# 1번 인덱스에 5,6을 삽입
np.insert(arr, 1, np.array([5,6]))

# arr[0:2]를 9로 수정
arr[0:2]=9

# 1, 2번 인덱스 요소 삭제
np.delete(arr,[1,2])


#%% 다양한 연산

arr1+1
arr1 + arr2

np.maximum(arr1, arr2)
np.minimum(arr1, arr2)

# 행렬의 행과 열을 바꾸는 전치 연산 - 전치행렬
arr3 = np.arange(0,4).reshape(2,2)
np.transpose(arr3)
arr3.T


#%% 두 2차 배열 통합

arr1 = np.arange(1, 7).reshape(2,3)
arr2 = np.arange(1, 7).reshape(2,3)
arr3 = np.arange(1, 7).reshape(2,3)

# 행방향 세로
test=np.append(arr1, arr2, axis=0)
np.concatenate((arr1, arr2), axis=0)

# 열방향 가로
np.append(arr1, arr2, axis=1)
np.concatenate((arr1, arr2), axis=1)

# 1차로 통합해버림 (axis가 없기 때문)
np.append(arr1, arr2)
np.concatenate((arr1, arr2))

#%% NA = null = None = np.nan = 결(불)측치=미정치
import numpy as np
arr = np.array([
    [1.6, np.nan],
    [3, None]
    ], dtype=float)
# 결(불)측지값 boolean 배열
np.isnan(arr)
# 결(불)측지값 0으로 수정
arr[np.isnan(arr)]=0 #기존값이 바뀜(인플레이스)
np.nan_to_num(arr, nan=0) #기존값을 바꾸지 않음

# 결(불)측지값 제거
arr[np.isnan(arr)==False]
arr[~np.isnan(arr)] #numpy에서 부정을 나타낼 때는 '~'를 쓴다


arr = np.array([
    [1.6, np.nan],
    [3, None]
    ], dtype=float)
# 각 행마다 결(불)측지값이 하나라도 존재하면 True, 아니면 False
np.isnan(arr).any(axis=1)

# 결(불)측지값 인덱스
nan_idx = np.where(np.isnan(arr))

# 결(불)측지값이 존재하는 행 삭제
np.delete(arr, nan_idx[0], axis=0)

# 결론: 수정을 권고 (삭제 x)

# 결(불)측지값이 존재하는 열 삭제
np.delete(arr, nan_idx[1], axis=0)


#%% sum

import numpy as np
arr = np.array([
    [1.6, np.nan],
    [3, None]
    ], dtype=float)

#전체계산
np.isnan(arr).sum()
#열마다 계산
np.isnan(arr).sum(axis=0)
#행마다 계산
np.isnan(arr).sum(axis=1)

#%% 통계합수 활용해서 집계

import numpy as np
lst = [
       [1, 2, 3],
       [5, 4, 6],
       [7, 8, 9]
       ]
arr = np.array(lst)
arr.sum()
arr.sum(axis=0)
arr.sum(axis=1)

#평균
arr.mean()
np.mean(arr)

np.var(arr)#분
np.std(arr)#표준편차
np.median(arr)#중간값
np.min(arr)
np.argmin(arr,axis=1)#최소값 index배열

arr = np.array([
    [1.6, np.nan],
    [3, None]
    ], dtype=float)

#null이 있으면 계산에 포함돼서 null이 나옴
np.var(arr,axis=0)

#가장 많이 존재하는요소 요소반환,갯수반환
from scipy import stats
a=np.array([1,1,3,4])
stats.mode(a)

#%% 정렬

import numpy as np

arr = np.array([-0.21082527, -0.0396508 , -0.75771892, -1.9260892 , -0.18137694,
-0.44223898, 0.32745569, 0.16834256])

np.sort(arr)#정렬
np.sort(arr)[::-1]#내림차순

arr2d = np.array([[-1.25627232, 1.65117477, -0.04868035],
                [ 0.7405744 , -0.67893699, -0.28428494],
                [ 0.02640821, -0.29027297, 0.34441534],
                [ 0.68394722, 0.26180229, 0.76742614],
                [ 1.00806827, 0.77977295, -1.36273314]])

np.sort(arr2d,axis=0)#열정렬
np.sort(arr2d,axis=1)#행정렬

#상위40% 출력
arr=np.array([[1, 0, 3],
              [4, 5, 6],
              [9, 8, 8]])
arr = arr.flatten() #1차원배열로 변환
five_idx=int(0.4*len(arr)) #상위40%의 갯수 저장

th=np.sort(arr)[::-1][five_idx] #내림차순으로 정렬후 가장 낮은값 검색
result = arr[arr>=th] #가장 낮은값보다 높은 요소 배열로 저장
np.unique(result) #중복제거후 정렬


#%% 퀴즈

import numpy as np

#20개 정수난수 1차 배열을 생성하고 최소,평균,최대와 표준편차를 구해본다
#그리고 상위 30%이내에 해당하는 값들 추출후 중복된 성분은 제거한다. 

arr=np.random.randint(1, 100, size=20)
min=np.min(arr)
max=np.max(arr)
std=np.std(arr)
avg=np.average(arr)
np.unique(arr[arr>=np.sort(arr)[-int(np.size(arr)*0.3)]])

#%%
import numpy as np
#school_2019.csv
arr=np.array([1,2,3])
np.savetxt('arr.csv',[arr],delimiter=',',fmt='%d',
           encoding='utf-8',header='col1,col2,col3',comments='')

arr=np.array([[1.2345,2,3],[4,5,6]])
#파일명,내용,구분자,타입,인코딩,헤더,코멘트
np.savetxt('arr_matrix.csv',arr,delimiter=',',fmt='%.2f',
           encoding='utf-8',header='col1,col2,col3',comments='')
help(np.savetxt)



#파일 읽기
#파일명,구분자,타입,인코딩,건너뛸줄(헤더에 쓰는데 #기호 있으면 불필요),사용할 칼럼
arr3 =np.loadtxt('arr.csv',delimiter=",",dtype=np.int64,
                 encoding='utf-8',skiprows=1)

arr3 = np.loadtxt('arr_matrix.csv',encoding='utf-8',
                  delimiter=',',skiprows=1,usecols=1)
#%%
import numpy as np

#지역,학교명,학급수,학생수,교사수
#최대 학급수의 초등학교,최대 학생수의 초등학교
#최대 교사수의 초등학교 출력
#복잡한 for 같은 제어로직 사용X 넘파이 사용

list_data =np.loadtxt('school_2019.csv',delimiter=",",dtype=np.str_,
                 encoding='utf-8',skiprows=1)
list_data[0]
list_data[:5,:]
list_data.shape
np.isnan(list_data)
data=list_data[:,2:].astype(np.int64)

max_index=np.argmax(data,axis=0)
#학급수
list_data[np.argmax(data[:,0]),1]
max_class=list_data[max_index[0],1]
num_class=list_data[max_index[0],2]
#학생수
list_data[np.argmax(data[:,1]),1]
max_student=list_data[max_index[1],1]
num_student=list_data[max_index[1],3]
#교사수
list_data[np.argmax(data[:,2]),1]
max_teacher=list_data[max_index[2],1]
num_teacher=list_data[max_index[2],4]

print(f'최대 학급수의 초등학교 : {max_class}, 학급수 : {num_class}')
print(f'최대 학생수의 초등학교 : {max_student}, 학생수 : {num_student}')
print(f'최대 교사수의 초등학교 : {max_teacher}, 교사수 : {num_teacher}')

#%% 퀴즈

#학급당 학생수(학생 /학급수)와 교사1인당 학생수(학생수/교사수)를 출력한다.
#단 / 는 나눗셈을 의미하고 학급당 학생수, 교사 1인당 학생수를 
#원배열(data)의 2열과 4열에 삽입하여 배열을 출력한다.
#만약 반올림 처리가 필요한 경우 np.round() 활용

list_data=np.insert(list_data,2,np.round(data[:,1]/data[:,0],1),axis=1)
list_data=np.insert(list_data,4,np.round(data[:,1]/data[:,2],1),axis=1)
print(list_data)

#과밀학급인 학교 행을 출력한다.
#단 과밀조건은 학급당 학생수 30인 이상
print(list_data[np.where(list_data[:,2].astype(np.float64)>30)])

