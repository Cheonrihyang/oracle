# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:14:53 2024

@author: ORC
"""

#%% 데이터가공

import csv
f=open('month_temp.csv','r',encoding=('utf-8'))
rdr = csv.reader(f)

#for line in rdr:
#    if line[1]=='2019-10-20':
#        print(f'평균기온{line[2]} 최저기온{line[3]} 최고기온{line[4]}')
sum=0
for line in rdr:
    if line[1][-2:]<='10':
        print(line)
        sum+=float(line[2])
avg=sum/10
f.close()

#%% 데이터가공

import csv
f=open('month_temp.csv','r',encoding=('utf-8'))
rdr = csv.reader(f)
next(rdr)
for line in rdr:
    a = round(float(line[4])-float(line[3]),2)
    print(f'{line[1]} {a}')
f.close()

#%% 데이터가공

import csv
from datetime import datetime
from dateutil.relativedelta import relativedelta
f=open('pharm_2019.csv','r',encoding=('utf-8'))
rdr = csv.reader(f)
next(rdr)

#for line in rdr:
#    if line[1]=='경주시' and line[0]=='신대원약국':
#        print(line[0],line[1],line[2],sep='/')

#for line in rdr:
#    if '로얄스포츠' in line[2] and '용인시 수지구' in line[2]:
#        print(line[0])

#count=1
#for line in rdr:
#    if int(line[3][:4])==2010 and '경상북도' in line[2]:
#        print(f'{count} {line[0]} {line[1]} {line[2]} {line[4]} {line[5]}')
#        count+=1
#f.close()

count=0
day = datetime.today()-relativedelta(years=5)
day = datetime.strftime(day,"%Y%m%d")
for line in rdr:
    if int(line[3]) >= int(day) and '용인시' in line[2]:
        count+=1
print(f'최근 용인시 5년 이내 개설 약국 수 : {count}개')

#%%

