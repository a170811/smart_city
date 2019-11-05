#!/usr/bin/env python

from bs4 import BeautifulSoup
import json
import pandas as pd
import requests
import time
from urllib import parse
from tqdm import tqdm

from stList import stList


stations = ['C0X110', 'C0O950', 'C0X120', 'C0X310', 'C0X280', 'C0X290']
# 南區 安南 麻豆 七股 將軍 北門

for year in tqdm(range(2010, 2020)):
    data_array = []
    for station in stations:
        for month in range(1, 13):
            if 2019 == year and month > 9:
                continue
            year = f'{year}'
            month = f'0{month}' if 10 > month else f'{month}'

            url = f'https://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain&station={station}&stname={parse.quote(parse.quote(stList[station][0]))}&datepicker={year}-{month}'

            try:
                response = requests.get(url)
            except:
                print('    Failed downloading: %s' % url)
                continue

            html = response.content
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find_all(attrs={'id': 'MyTable'})[0]

            for i, row in enumerate(table.findAll('tr')):
                if 1 == i:
                    cols = [ele.text.replace('\xa0', '') for ele in row.findAll('th')]
                elif 2 < i:
                    data = [ele.text.replace('\xa0', '') for ele in row.findAll('td')]
                    data[0] = f'{year}-{month}-{data[0]}'
                    data_array.append(data)

    df = pd.DataFrame(data_array, columns=cols)
    df = df.replace('T', 'NaN')
    df = df.replace('X', 'NaN')
    df = df.replace('/', 'NaN')
    df = df.replace('...', 'NaN')

    df[['測站氣壓(hPa)', '海平面氣壓(hPa)', '測站最高氣壓(hPa)', '測站最低氣壓(hPa)', '氣溫(℃)', '最高氣溫(℃)', '最低氣溫(℃)', '露點溫度(℃)', '相對溼度(%)', '最小相對溼度(%)', '風速(m/s)', '風向(360degree)', '最大陣風(m/s)', '最大陣風風向(360degree)', '降水量(mm)', '降水時數(hour)', '最大十分鐘降水量(mm)', '最大六十分鐘降水量(mm)', '日照時數(hour)', '日照率(%)', '全天空日射量(MJ/㎡)', '能見度(km)', 'A型蒸發量(mm)', '日最高紫外線指數', '總雲量(0~10)']] =\
        df[['測站氣壓(hPa)', '海平面氣壓(hPa)', '測站最高氣壓(hPa)', '測站最低氣壓(hPa)', '氣溫(℃)', '最高氣溫(℃)', '最低氣溫(℃)', '露點溫度(℃)', '相對溼度(%)', '最小相對溼度(%)', '風速(m/s)', '風向(360degree)', '最大陣風(m/s)', '最大陣風風向(360degree)', '降水量(mm)', '降水時數(hour)', '最大十分鐘降水量(mm)', '最大六十分鐘降水量(mm)', '日照時數(hour)', '日照率(%)', '全天空日射量(MJ/㎡)', '能見度(km)', 'A型蒸發量(mm)', '日最高紫外線指數', '總雲量(0~10)']].astype(float)
    # 有轉成int 或 float 的才被平均


    df = df.rename(columns = {'觀測時間(day)': 'date'})
    df = df.groupby('date').mean()
    # skip NaN, 如果全部的直都是 NaN，則平均就是NaN
    # NaN 存成 csv 就是空的

    df.to_csv(f'data/weather_{year}.csv')
    print(f'data/weather_{year}.csv')

# https://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain&station=467410&stname=%25E8%2587%25BA%25E5%258D%2597&datepicker=2010-09

