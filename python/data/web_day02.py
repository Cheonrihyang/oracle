# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:45:38 2024

@author: ORC
"""

#%% BeautifulSoup

#requests 사용 경우
import requests, bs4

resp = requests.get('http://finance.naver.com/')
resp.status_code
#resp.raise_for_status()
resp.encoding='euc-kr'
html=resp.text

#BeautifulSoup 생성
soup = bs4.BeautifulSoup(html,'html.parser')

#find(태그)
#조건에 맞는 첫번째 태그를 하나 가져옴
soup.find('title').text
soup.find('title').string

#find_all(태그)
#조건에 맞는 모든 태그를 list로 가져옴
lst = soup.find_all('a')
#디폴트가 all이라 동일 결과
soup('a')
lst[1].text
lst[0]
#검색 후 인덱싱
lst[0]['href']

#속성명 검색
soup.find(id='start')
#태그명 속성명 동시 검색
soup.find('div',id='start')



#urllib 사용 경우
import urllib.request as req
resp = req.urlopen("https://finance.naver.com/")
resp.status
resp.getheader('content-type')

html_byte = resp.read()
html=html_byte.decode('euc-kr')
soup = bs4.BeautifulSoup(html,'html.parser')


#%% select(css 선택)

#주요뉴스 첫번째 뉴스 크롤링
s = '#content > div.article > div.section > div.news_area._replaceNewsLink > div > ul > li:nth-child(1) > span > a'
soup.select_one(s)

#주요뉴스 제목 크롤링
s = '#content > div.article > div.section > div.news_area._replaceNewsLink > div > ul > li > span > a'
news_lst = soup.select(s)
news_lst[5].text
news_lst[5].getText()
for i,news in enumerate(news_lst):
    print(i+1,news.text)
    print('-'*50)


s = '#content > div.article > div.section > div.news_area._replaceNewsLink > div > ul > li > span > a[href*="/news/news_read.naver"]'
news_lst = soup.select(s)

#%% 퀴즈

#네이버 시장지표의 미국 USD 환률과 주요뉴스제목 10개를 출력하라.
import requests, bs4
resp = requests.get('https://finance.naver.com/marketindex/')
resp.encoding='euc-kr'

html=resp.text
soup = bs4.BeautifulSoup(html,'html.parser')

s = '#content > div.section_news > div > ul >li>p'
lst=soup.select(s)
for text in lst:
    print(text.text)
    print('-'*50)
    
    
print(soup.find_all('span',class_="value")[0].text,"원")


resp = requests.get('https://finance.naver.com/marketindex/exchangeList.naver')
resp.encoding='euc-kr'

html=resp.text
soup = bs4.BeautifulSoup(html,'html.parser')


s = 'body > div > table > tbody > tr:nth-child(1) > td.sale'
soup.select_one(s).text

#%% 할리스커피 50개 매장 정보를 크롤링

import urllib.request as req , bs4
from itertools import count
#https://www.hollys.co.kr/   
#-> Store

def hollys_store(result):   
    for page in count():
        Hollys_url = 'https://www.hollys.co.kr/store/korea/korStore2.do?pageNo=%d&sido=&gugun=&store=' %(page + 1)
        html = req.urlopen(Hollys_url)
        soupHollys = bs4.BeautifulSoup(html, 'html.parser')
        tag_tbody = soupHollys.find('tbody')
        if "등록된 지점이 없습니다." in tag_tbody.text:
            break
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

#0부터 시작하는 숫자 무한 이터레이터 생성
from itertools import count

#count(시작수,증가수) 무한반복
for page in count(4,2):
    print(page)
    #if가 true라면 멈춤
    if page == 100:
        break
    
#%% 셀레니움

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

#브라우저 설정후 창 열기
chrome_options = Options()
driver = webdriver.Chrome(options=chrome_options)

#안정성을 위해 5초 대기
driver.implicitly_wait(time_to_wait=5)
driver.get("http://python.org")
print(driver.current_url)
print(driver.title)

driver.get("https://naver.com")
#뒤로가기
driver.back()
#앞으로가기
driver.forward()

driver.get("http://python.org")
from selenium.webdriver.common.by import By
#top > nav > ul > li.python-meta.current_item.selectedcurrent_branch.selected
#top > nav > ul > li.psf-meta
#top > nav > ul > li.docs-meta
#파이썬 홈페이지 메뉴의 공통점은 top > nav > ul > li

#menus = driver.find_elements(By.CSS_SELECTOR,'#top ul.menu li')
menus = driver.find_elements(By.CSS_SELECTOR,'#top > nav > ul > li')
pypi=None
for m in menus:
    if m.text == "PyPI": # PyPI 메뉴 찾기
        pypi = m
    print(m.text)
#찾은 메뉴 클릭
pypi.click()
print("PyPI 클릭완료")
driver.back()
#driver.quit()


#id가 search-field인 요소찾기
search_field = driver.find_element(By.CLASS_NAME,'search-field')
search_field.clear() #지우기
search_field.send_keys('numpy') #검색값 입력
print("검색값 삽입완료")

from selenium.webdriver.common.keys import Keys
search_field.send_keys(Keys.RETURN) #검색 실행
print("값 입력완료")
driver.back()

#id가 submit인 요소찾기
submit_btn = driver.find_element(By.ID,"submit")
search_field.clear()
search_field.send_keys('ID')
submit_btn.click()
print("id로 submit클릭완료")
driver.back()

#name이 submit인 요소찾기
submit_btn = driver.find_element(By.NAME,"submit")
search_field.clear()
search_field.send_keys('NAME')
submit_btn.click()
print("name으로 submit클릭완료")
driver.back()

#css스타일로 #submit 부르기
submit_btn = driver.find_element(By.CSS_SELECTOR,"#submit")
search_field.clear()
search_field.send_keys('CSS_SELECTOR')
submit_btn.click()
print("css스타일로 submit클릭완료")
driver.back()

#XPATH로 submit 부르기
submit_btn = driver.find_element(By.XPATH,'//*[@id="submit"]')
search_field.clear()
search_field.send_keys('XPATH')
submit_btn.click()
print("XPATH로 submit클릭완료")
driver.back()

# JS코드 기반
driver.execute_script("document.querySelector('#submit').click();")
print("JS로 submit클릭완료")

#%% 퀴즈

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
driver = webdriver.Chrome(options=chrome_options)

driver.implicitly_wait(time_to_wait=5)
driver.get("https://www.google.com/")

ip = driver.find_element(By.NAME,'q')
ip.send_keys('naver')

btn = driver.find_element(By.XPATH,'/html/body/div[1]/div[3]/form/div[1]/div[1]/div[4]/center/input[1]')
btn.click()


link2 = driver.find_element(By.XPATH,'//*[@id="rso"]/div[1]/div/div/div/div/div/div/div/div[1]/div/span/a')
print(link2.text[link2.text.index("http"):])
