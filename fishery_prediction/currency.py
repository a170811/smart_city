#!/usr/bin/env python


from datetime import datetime, timedelta
import json
from sys import argv
import time

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests


class Currency():
    def __init__(self, base, start, end):

        self.base = base
        self.start = datetime.strptime(start, '%Y-%m-%d')
        self.end = datetime.strptime(end, '%Y-%m-%d')
        self.days = (self.end - self.start).days + 1
        self.delta = timedelta(days=1)

    def get(self, targets):

        cur_res = pd.DataFrame(columns=['date']+targets)

        date = self.start
        for i in range(self.days):
            date_str = date.strftime('%Y-%m-%d')
            url = f"https://www.xe.com/zh-HK/currencytables/?from={self.base}&date={date_str}"

            response = requests.get(url)
            html = response.content
            soup = BeautifulSoup(html, 'html.parser')
            all_td = soup.find_all('td')
            td_res = [td.text for td in all_td]

            cur_res.loc[i] = [date_str] + self._get_currency(targets, td_res)
            date += self.delta
        return cur_res

    def _get_currency(self, abbrs, res, ctype='base_per_unit'):

        cur = []
        if 'unit_per_base' == ctype:
            pos = lambda x: x + 2
        elif 'base_per_unit' == ctype:
            pos = lambda x: x + 3
        else:
            raise Exception(f'Error, type `{ctype}` is not found')

        for abbr in abbrs:
            try:
                idx = res.index(abbr)
            except:
                idx = None
            cur.append(res[pos(idx)] if idx is not None else idx)
        return cur

if '__main__' == __name__:

    if len(argv) < 3:
        print(f'Useage: python3 {argv[0]} [start_date] [end_date]')
        exit()
    start = argv[1]
    end = argv[2]

    targets = ['USD', 'CAD', 'SAR', 'AED']
    # ['美金', '加拿大元', '沙烏地里亞爾', '阿聯迪拉姆']
    # 取 base per Unit

    c = Currency('TWD', start, end)
    res = c.get(targets)
    res.to_csv(f'./tmp/{start}_to_{end}.csv')
    print(f'{start} to {end}, Done')


