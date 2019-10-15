#!/usr/bin/env python

from bs4 import BeautifulSoup
import pandas as pd
import requests
import time
from urllib import parse
# from datetime import datetime, timedelta



# stList = {'100':['永康', '???'], '101':['台南', '???']}
# stations = ['100', '101']
# for station in stations:
#     for year in range(2010, 2020):
#         for month in range(1, 13):
#             month = f'0{month}' if 10 > month else month
#             url = f'https://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain&station={station}&stname={parse.quote(parse.quote(stList[station][0]))}&datepicker={year}-{month}'
#             print(url)
#
url = 'https://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain&station=466910&stname=%25E9%259E%258D%25E9%2583%25A8&datepicker=2019-09'

try:
    response = requests.get(url)
except:
    print('    Failed downloading: %s' % url)
month='01'
html = response.content

soup = BeautifulSoup(html, 'html.parser')
table = soup.find_all(attrs={'id': 'MyTable'})[0]

data_array = []
for i, row in enumerate(table.findAll('tr')):
    if 1 == i:
        cols = [ele.text.replace('\xa0', '') for ele in row.findAll('th')]
    elif 2 < i:
        data = [ele.text.replace('\xa0', '') for ele in row.findAll('td')]
        data[0] = month+data[0]

        data_array.append(data)

df = pd.DataFrame(data_array, columns=cols)
print(df)
