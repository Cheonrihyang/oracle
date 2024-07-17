# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:05:12 2024

@author: 82103
"""

#%% requests
# 웹페이지 탐색 수집 분류 지정
import requests #HTTP 요청 모듈
resp = requests.get("https://finance.naver.com/")
resp.status_code #200 - 정상
resp.encoding
# 응답 데이터를 문자열로 리턴하는 text 속성
# html 소스 보기
html = resp.text
print(html)

#get params
#query 파라미터 사전으로 요청
resp = requests.get('https://search.naver.com/search.naver', params={'query':'파이썬'})
html = resp.text
print(html)

# 응답을 파일로 저장
url = 'https://finance.naver.com/marketindex/'
resp = requests.get(url)

with open('marketindex.html', 'w', encoding='UTF-8') as f:
    f.write(resp.text)


# 응답결과가 바이너리(이미지)
resp = requests.get("https://blogimgs.pstatic.net/nblog/mylog/post/og_default_image_160610.png")
#response.content 이미지 데이터를 가지고 파일에다(f) 데이터를 씀(write)
with open('naver_blog_logo.png', 'wb') as f:
    f.write(resp.content)


    
#%% urllib

import urllib.request as req
resp = req.urlopen('https://finance.naver.com/')
resp.status
# charset 확인
resp.getheader('content-type')
# 인코딩 바이트 문자열
html_byte = resp.read()
# decode
html = html_byte.decode('euc-kr')
print(html)

# 파일로 응답 저장
with open('naverfinance.html', 'w', encoding='UTF-8') as f:
    f.write(html)

# 응답결과가 바이너리(이미지)
req.urlretrieve('https://blogimgs.pstatic.net/nblog/mylog/post/og_default_image_160610.png',
                'naver_blog_logo2.png')



#%% BeautifulSoup

# 1. urllib 사용할 경우
import bs4
import urllib.request as req
resp = req.urlopen('https://finance.naver.com/')
resp.status
# charset 확인
resp.getheader('content-type')
# 인코딩 바이트 문자열
html_byte = resp.read()
# decode
html = html_byte.decode('euc-kr')
soup = bs4.BeautifulSoup(html, 'html.parser')


######################################################################
# 2. request 사용할 경우
import	requests,	bs4
# 웹에있는 소스 가져오기
resp = requests.get('http://finance.naver.com/')
# resp.raise_for_status()
resp.encoding = 'euc-kr' 
html = resp.text

# BeautifulSoup 생성
soup = bs4.BeautifulSoup(html, 'html.parser')

# find(태그) - 해당 조건에 맞는 첫번째 태그를 가져온다
soup.find('title') #태그랑 같이 불러옴
soup.find('title').text #문자열만 가지고옴
soup.find('title').string #문자열만 가지고옴
soup.find('a').text

# find_all(태그) - 모든 태그들을 추출하여 list로 리턴
lst = soup.find_all('a')
lst[0].text
lst[1].text
lst[0]['href']

soup('a') #find_all이 기본값이라 이렇게도 사용 가능
soup('a')[0].text
# find(태그의 속성)
soup.find(id='start') #아이디명으로 검색

# 태그명과 속성으로 요소를 추출
soup.find('div', id='start')


#%% select(css 선택자)

s = '#content > div.article > div.section > div.news_area._replaceNewsLink > div > ul > li:nth-child(6) > span > a'
# 주요뉴스의 탑(첫번째) 뉴스 제목
soup.select_one(s)

# '주요뉴스' 라는 제목
s = '#content > div.article > div.section > div.news_area._replaceNewsLink > div'
soup.select(s)

# '주요뉴스' 라는 제목 불필요한 정보들 제거
s = '#content > div.article > div.section > div.news_area._replaceNewsLink > div > ul'
news_lst = soup.select(s)
news_lst[0]


# '주요뉴스' 라는 제목 6개 목록
# li 태그의 :nth-child() 삭제
s = "#content > div.article > div.section > div.news_area._replaceNewsLink > div > ul > li > span > a"
news_lst = soup.select(s)
news_lst[0].text
news_lst[1].text
news_lst[1].getText()

# enumerate(all_element) : (요소인덱스, 요소)를 리턴
for i, news in enumerate(news_lst):
    print(i+1, news.text)
    print('-'*50)

# 속성 선택자
s = "#content > div.article > div.section > div.news_area._replaceNewsLink > div > ul > li > span > a[href*=/'news/news_read.naver?']"
news_lst = soup.select(s)


#%% 퀴즈
# 네이버 시장지표의 미국 USD 환율과 주요뉴스제목 10개를 출력
import bs4
import urllib.request as req
resp = req.urlopen('https://finance.naver.com/marketindex/')
resp.status
# charset 확인
resp.getheader('content-type')
# 인코딩 바이트 문자열
html_byte = resp.read()
# decode (BeautifulSoup 자체에서 decoding 해줌. 안해줘도 상관 없으나 해주는걸 권장)
html = html_byte.decode('euc-kr')
soup = bs4.BeautifulSoup(html, 'html.parser')
# soup = bs4.BeautifulSoup(resp, 'html.parser') 로 바로 대입해줘도 알아서 디코딩해줌
# 환율 주요뉴스제목
s = "#content > div.section_news > div > ul > li > p"
news_lst = soup.select(s)
for i, news in enumerate(news_lst):
    print(i+1, news.text)
    print('-'*50)

# 미국 USD 환율
rate = "#exchangeList > li.on > a.head.usd > div > span.value"
rate_usd = soup.select(rate)
print(rate_usd[0].text, "원")


#%% 할리스커피 크롤링

import urllib.request as req , bs4
def hollys_store(result):   
    for page in range(1,6): # 반복횟수 5번 (페이지 당 10개의 매장 존재)
        #1페이지 pageNo=1 https://www.hollys.co.kr/store/korea/korStore2.do?pageNo=1&sido=&gugun=&store=  
        #pageNo=%d가 %page
        Hollys_url = 'https://www.hollys.co.kr/store/korea/korStore2.do?pageNo=%d&sido=&gugun=&store=' %page
        print(Hollys_url)
        html = req.urlopen(Hollys_url)
        soupHollys = bs4.BeautifulSoup(html, 'html.parser')
        tag_tbody = soupHollys.find('tbody') # 매장정보(행) 10개
        for store in tag_tbody.find_all('tr'):            
            store_td = store.find_all('td')
            store_name = store_td[1].text
            store_sido = store_td[0].text
            store_address = store_td[3].text
            store_phone = store_td[5].text
            #데이터프레임에서 하나의 행에 네개의 컬럼이 생성
            result.append([store_name]+[store_sido]+[store_address]
                          +[store_phone])
           
    return

def main():
    result = []
    print('Hollys store crawling >>>>>>>>>>>>>>>>>>>>>>>>>>')
    hollys_store(result)       
    print(result[:])
       
main()

#%% count()
# 0부터 시작하는 숫자 무한 이터레이터 생성
from itertools import count
for page in count(4, 2): #count(시작 수, 간격)
    print(page)
    if page == 100:
        break


#%% 할리스커피 매장 정보 전체 크롤링
# 총매장개수 모르는 경우

import urllib.request as req , bs4
from itertools import count
def hollys_store2(result):#result은 이차리스트    
    for page in count():          
        Hollys_url = 'https://www.hollys.co.kr/store/korea/korStore2.do?pageNo=%d&sido=&gugun=&store=' %(page + 1)
        print(Hollys_url)
        #time.sleep(1)
        html = req.urlopen(Hollys_url)
        soupHollys = bs4.BeautifulSoup(html, 'html.parser')
        ##contents > div.content > fieldset > fieldset > div.tableType01 > table > tbody  
        tag_tbody = soupHollys.find('tbody')
       
        for store in tag_tbody.find_all('tr'):
            print('>>>>' , store.text)
            try:                
                store_td = store.find_all('td')            
                 # 탈출조건: td 인덱스 존재하지않으면(인덱스 예외) 함수 탈출
                store_name = store_td[1].text
                store_sido = store_td[0].text
                store_address = store_td[3].text
                store_phone = store_td[5].text                
                
                result.append([store_name]+[store_sido]+[store_address]
                              +[store_phone])
            except:
                print('더이상 매장정보가 없다')                
                return #함수 탈출    

def main():
    result = []
    print('Hollys store crawling >>>>>>>>>>>>>>>>>>>>>>>>>>')
    hollys_store2(result)      
    print(result[:])
    print('검색된 매장수 : %d' % len(result))
       
main()


#%% selenium

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# 브라우저 열기
# 조금만 기다리면 selenium으로 제어할 수 있는 브라우저 새 창이 뜬다
chrome_options = Options()
driver = webdriver.Chrome(options=chrome_options)

# 5초간 대기
# 브라우저가 get() 이후 페이지로딩(파싱)될 때까지 5초 대기
driver.implicitly_wait(time_to_wait=5)

# 웹페이지 가져오기
driver.get("http://python.org")
print(driver.current_url) #창주소
print(driver.title) #창제목
###########################################
driver.get("http://www.naver.com")
driver.back() #뒤로
driver.forward() #앞으로
###########################################
# 웹페이지 가져오기
driver.get("http://python.org")

from selenium.webdriver.common.by import By
#top > nav > ul > li.python-meta.current_item.selectedcurrent_branch.selected
#top > nav > ul > li.psf-meta
#top > nav > ul > li.docs-meta
# 반복규칙: top > nav > ul > li

################################################################
# WebElement 객체들의 리스트
# menus = driver.find_elements(By.CSS_SELECTOR, '#top > nav > ul.menu > li > a')
menus = driver.find_elements(By.CSS_SELECTOR, '#top > nav > ul > li')
# PyPI 요소 찾기
pypi = None
for m in menus:
    if m.text == "PyPI":
        pypi = m
    print(m.text)
print(m)
pypi.click()

# 브라우저 종료
driver.quit()


chrome_options = Options()
driver = webdriver.Chrome(options=chrome_options)
driver.implicitly_wait(time_to_wait=5)
driver.get("http://python.org")

# 입력 필드 사용자의 입력 받기

# class_name 으로 찾기
search_field = driver.find_element(By.CLASS_NAME, 'search-field')

# Id 로 찾기
go_field = driver.find_element(By.ID, 'submit')

# Name 으로 찾기
go_field = driver.find_element(By.NAME, 'submit')

# 입력 필드 내용 삭제
search_field.clear()
# 검색어 입력
search_field.send_keys('numpy')

# 엔터 입력
from selenium.webdriver.common.keys import Keys
search_field.send_keys(Keys.RETURN)

go_field.click()

# Id
driver.back()
go_field = driver.find_element(By.ID, 'submit')
go_field.click()

# Name
driver.back()
go_field = driver.find_element(By.NAME, 'submit')
go_field.click()

# CSS_Selector
driver.back()
go_field = driver.find_element(By.CSS_SELECTOR, '#submit')
go_field.click()

# XPATH 
driver.back()
go_field = driver.find_element(By.XPATH, '//*[@id="submit"]')
go_field.click()

# JS코드 기반 검색창의 버튼클릭
# execute_script(JS코드)
driver.execute_script("document.querySelector('#submit').click();")


#%% 퀴즈
# 구글로 numpy 검색
# 검색결과리스트에서 첫번째 결과 링크가져오기

# import
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

chrome_options = Options()
driver = webdriver.Chrome(options=chrome_options)

driver.implicitly_wait(3)

# 구글로 이동
driver.get("https://www.google.com/")

# XPATH 이용하여 검색하기
search_field = driver.find_element(By.XPATH, '//*[@id="APjFqb"]')
search_field.clear()
search_field.send_keys('numpy')

search_field.send_keys(Keys.RETURN)

# 첫번째 검색결과의 링크에서 주소 추출
first_result = driver.find_element(By.XPATH, '//*[@id="rso"]/div[1]/div/div/div/div/div/div/div/div[1]/div/span/a')
first_link = first_result.get_attribute('href')
# 첫번째 검색결과로 이동
first_result.click()
print(first_link)

# 종료
driver.quit()


#%% BeautifulSoup과 통합

time.sleep(1)

# 현재페이지 page_source
html = driver.page_source

# 현재페이지를 BeautifulSoup과 통합
# 이렇게 req에 소스를 저장했으면 이 req가 HTML parser를 사용해야 한다
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')


# header = soup.select('#the-python-tutorial > h1')
# header[0].text
# header = soup.select_one('#the-python-tutorial > h1')
# header.text

# numpy 로고 우측의 문구
header = soup.select_one('body > section.hero > div > div > div > div.flex-column > div.hero-subtitle')

print(header.text)



#%% 실시간 검색어 10개 출력(selenium 기반)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

chrome_options = Options()
driver = webdriver.Chrome(options=chrome_options)

driver.implicitly_wait(3)

# 시그널 실시간검색어로 이동
driver.get("https://www.signal.bz/")

html = driver.page_source

####################################################################
# 강사님 방식
# 두개의 :nth-child 제거
url = '#app > div > main > div > section > div > section > section:nth-child(2) > div:nth-child(2) > div > div > div > a > span.rank-text'
results = driver.find_elements(By.CSS_SELECTOR, url)

lst = []
for res in results:
    lst.append(res.text)
    
print('실시간 검색어', lst)

# 강사님 방식을 리스트 내포로
lst = [res.text for res in results]
print('실시간 검색어', lst)


#%% 실시간 검색어 10개 출력(BeautifulSoup 통합)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
driver = webdriver.Chrome(options=chrome_options)

driver.implicitly_wait(3)

# 시그널 실시간검색어로 이동
driver.get("https://www.signal.bz/")
html = driver.page_source

# BeautifulSoup
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')

trending_keywords = []
# soup.select_one 이용하여 실시간 검색어 찾기
for i in range(1, 6):
    url = f'#app > div > main > div > section > div > section > section:nth-child(2) > div:nth-child(2) > div > div:nth-child(1) > div:nth-child({i}) > a > span.rank-text'
    trending = soup.select_one(url)
    trending_keywords.append(trending.text)
    
for i in range(1, 6):
    url = f'#app > div > main > div > section > div > section > section:nth-child(2) > div:nth-child(2) > div > div:nth-child(2) > div:nth-child({i}) > a > span.rank-text'
    trending = soup.select_one(url)
    trending_keywords.append(trending.text)

for idx, keyword in enumerate(trending_keywords, start=1):
    print(f'{idx} {keyword}')

####################################################################
# 리스트 내포 사용하기
urls = [f'#app > div > main > div > section > div > section > section:nth-child(2) > div:nth-child(2) > div > div:nth-child(1) > div:nth-child({i}) > a > span.rank-text' for i in range(1,6)] + \
    [f'#app > div > main > div > section > div > section > section:nth-child(2) > div:nth-child(2) > div > div:nth-child(2) > div:nth-child({i}) > a > span.rank-text' for i in range(1,6)]

trending_keywords = [soup.select_one(url).text for url in urls]

for idx, keyword in enumerate(trending_keywords, start=1):
    print(f'{idx} {keyword}')