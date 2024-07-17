# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:05:22 2024

@author: ORC
"""

#%% 웹 크롤링

import requests

resp=requests.get("https://finance.naver.com/")
#현재상태
resp.status_code
resp.encoding

#html 소스보기
html = resp.text
print(html)

#검색한 소스 저장
resp=requests.get("https://search.naver.com/search.naver",params={'query':'파이썬'})
resp.encoding
html = resp.text
print(html)

#결과 저장
url = 'https://finance.naver.com/marketindex/'
resp = requests.get(url)

with open("marketindex.html","w",encoding=('utf-8')) as f:
    f.write(resp.text)

#이미지 크롤링
resp = requests.get("https://blogimgs.pstatic.net/nblog/mylog/post/og_default_image_160610.png")
#response.content이미지 데이터를 가지고 파일에다(f) 데이터를 씀(write)
#wb(바이너리)를 사용하고 content를 사용함.
with open('naver_blog_logo.png', 'wb') as f:
    f.write(resp.content)

#%% urllib

import urllib.request as req
resp = req.urlopen("https://finance.naver.com/")
resp.status
resp.getheader('content-type')

#인코딩 바이트 문자
html_byte = resp.read()
html=html_byte.decode('euc-kr')
print(html)

#결과 저장
with open("finance.html","w",encoding=('utf-8')) as f:
    f.write(html)
    
#이미지 크롤링
resp = req.urlretrieve("https://blogimgs.pstatic.net/nblog/mylog/post/og_default_image_160610.png")
with open('naver_blog_logo.png', 'wb') as f:
    f.write(resp.content)