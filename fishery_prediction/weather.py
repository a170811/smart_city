#!/usr/bin/env python

from bs4 import BeautifulSoup
import requests
import time
# from datetime import datetime, timedelta



url = 'https://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain&station=466910&stname=%25E9%259E%258D%25E9%2583%25A8&datepicker=2019-09'

try:
    response = requests.get(url)
except:
    print('    Failed downloading: %s' % url)

html = response.content

soup = BeautifulSoup(html, 'html.parser')
table = soup.find_all(attrs={'id': 'MyTable'})[0]

data_array = []
for i, row in enumerate(table.findAll('tr')):
    if 1 == i:
        col_name = [ele.text.replace('\xa0', '') for ele in row.findAll('th')]

    data = [ele.text.replace('\xa0', '') for ele in row.findAll('td')]

    data_array.append(data)

print(col_name)
print(data_array)
