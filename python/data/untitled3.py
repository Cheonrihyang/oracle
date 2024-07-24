# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:42:32 2024

@author: ORC
"""

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

#%%

import urllib.request as req , bs4
from itertools import count

