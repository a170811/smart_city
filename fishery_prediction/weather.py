#!/usr/bin/env python

from bs4 import BeautifulSoup
import json
import pandas as pd
import requests
import time
from urllib import parse

from stList import stList


stations = ['467410', '467420']

data_array = []
for station in stations:
    for year in range(2009, 2020):
        for month in range(1, 13):
            year = f'{year}'
            month = f'0{month}' if 10 > month else f'{month}'

            url = f'https://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain&station={station}&stname={parse.quote(parse.quote(stList[station][0]))}&datepicker={year}-{month}'

            try:
                response = requests.get(url)
            except:
                print('    Failed downloading: %s' % url)

            html = response.content
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find_all(attrs={'id': 'MyTable'})[0]

            for i, row in enumerate(table.findAll('tr')):
                if 1 == i:
                    cols = [ele.text.replace('\xa0', '') for ele in row.findAll('th')]
                elif 2 < i:
                    data = [ele.text.replace('\xa0', '') for ele in row.findAll('td')]
                    data[0] = year+month+data[0]

                    data_array.append(data)

df = pd.DataFrame(data_array, columns=cols)
df.to_csv('weather_2009~2019.csv')

# https://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain&station=467410&stname=%25E8%2587%25BA%25E5%258D%2597&datepicker=2010-09

