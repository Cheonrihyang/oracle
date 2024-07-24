# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:56:50 2024

@author: ORC
"""

#%% 다중위치,마커의 MarkerCluster
import pandas as pd
import folium

bike=pd.read_excel('공공자전거 대여소 정보_201905.xlsx')
#bike.rename(columns={'위도':'lat','경도':'lng'}, inplace=True)

#맵의 MarkerCluster
#1. folium Map 생성 위도,경도 평균
bmap = folium.Map(location=[bike['위도'].mean(), bike['경도'].mean()], zoom_start=11)

#마커들을 마커배열안에 넣는 과정
#2. marker_cluster 변수는 마커들의 배열 생성하고 folium Map에 추가된  MarkerCluster의 Map
from folium.plugins import MarkerCluster
marker_cluster = MarkerCluster().add_to(bmap)

#3. 마커를 생성하고 marker_cluster에 추가
#모든 거치대수를 문자형으로 변환 

#bike['거치대수']가 정수이므로 + 결합을 위해서 문자열로 변환
bike['거치대수']=[str(value) for value in bike['거치대수']]

#반복문으로 데이터를 기반으로 마커를 생성
#마커를 marker_cluster에 추가

for n in bike.index:  #n 인덱스번호
        folium.Marker([bike['위도'][n], bike['경도'][n]],
                      popup='거치대수 :' + bike['거치대수'][n],
                     tooltip=bike['대여소명'][n]).add_to(marker_cluster)
'''
location = list(zip(bike['위도'],bike['경도']))
for n in bike.index:  
        folium.Marker(location[n],
     popup='거치대수 :' + bike['거치대수'][n],
                     tooltip=bike['대여소명'][n]).add_to(marker_cluster)
'''          
bmap.save('mapClusterMarker.html') #mapClusterMarker.html 생성
   

