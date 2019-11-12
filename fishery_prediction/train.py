#!/usr/bin/env python3

from sys import argv

import numpy as np
import pandas as pd

from build_set import RowDataHandler, preprocess
from model import train_and_eval_model, linear_regression

def main():# {{{

    path = './data/use'

    # value is None for daily data
    data_common = {
        'currency': None,
        'powder_feed': ['same', 'same'],
        'yellow_bean': ['same'],
        'weather': None
    }
    data_wu = {
        'wu_export': ['divide', 'same'],
        'wu_price_perDate': None,
        # 'wu_price_perMonth': ['same', 'divide'],
    }
    data_chi = {
        'chi_export': ['divide', 'same'],
        'chi_price_perDate': None,
        'chi_price_perMonth': ['same', 'divede'],
        'chi_small_fish': ['divide', 'same']
    }

    d = RowDataHandler()
    for filename, inpu_method in data_common.items():
        d.add(pd.read_csv(f'{path}/{filename}.csv'), inpu_method)
    for filename, inpu_method in data_wu.items():
        d.add(pd.read_csv(f'{path}/wu/{filename}.csv'), inpu_method)

    # ['date', 'USD', 'CAD', 'SAR', 'AED', 'pf_weight', 'pf_price', 'yb_price', 'temp', 'temp_high', 'temp_low', 'point_temp', 'wu_ex_weight', 'wu_ex_price', 'wu_day_price', 'wu_day_amount']
    merged_data = d.get_data(*d.get_start_end_tick())[['date', 'wu_day_price']]
    data, time = preprocess(merged_data, 1, 1, 7, 1, ans_col='wu_day_price')
    linear_regression(data['test_x'], data['test_y'])

    merged_data = d.get_data(*d.get_start_end_tick())
    merged_data = merged_data[['date', 'temp', 'temp_high', 'temp_low', 'wu_ex_weight', 'wu_ex_price', 'wu_day_price', 'wu_day_amount']]
    merged_data = merged_data.dropna().reset_index(drop=True)
    data, time = preprocess(merged_data, 1, 1, 7, 1, ans_col='wu_day_price')
    train_and_eval_model('weather_7dl', 'large', **data)

    merged_data = d.get_data(*d.get_start_end_tick())
    merged_data = merged_data[['date', 'temp', 'temp_high', 'temp_low', 'wu_ex_weight', 'wu_ex_price', 'wu_day_price', 'wu_day_amount']]
    merged_data = merged_data.dropna().reset_index(drop=True)
    data, time = preprocess(merged_data, 1, 1, 14, 1, ans_col='wu_day_price')
    train_and_eval_model('weather_14dl', 'large', **data)

    merged_data = d.get_data(*d.get_start_end_tick())
    merged_data = merged_data[['date', 'temp', 'temp_high', 'temp_low', 'wu_ex_weight', 'wu_ex_price', 'wu_day_price', 'wu_day_amount']]
    merged_data = merged_data.dropna().reset_index(drop=True)
    data, time = preprocess(merged_data, 7, 7, 4, 1, ans_col='wu_day_price')
    train_and_eval_model('weather_4wl', 'large', **data)

    merged_data = d.get_data(*d.get_start_end_tick())
    merged_data = merged_data[['date', 'temp', 'temp_high', 'temp_low', 'wu_ex_weight', 'wu_ex_price', 'wu_day_price', 'wu_day_amount']]
    merged_data = merged_data.dropna().reset_index(drop=True)
    data, time = preprocess(merged_data, 30, 30, 4, 1, ans_col='wu_day_price')
    train_and_eval_model('weather_4ml', 'large', **data)
# }}}


def exp1():# {{{
    # data before 2013 to drop or not

    d = RowDataHandler()
    for filename, inpu_method in data_common.items():
        d.add(pd.read_csv(f'{path}/{filename}.csv'), inpu_method)
    for filename, inpu_method in data_wu.items():
        d.add(pd.read_csv(f'{path}/wu/{filename}.csv'), inpu_method)

    start, end = d.get_start_end_tick()
    merged_data = d.get_data(start, end)
    data_retain, _ = preprocess(merged_data, 1, 1, 7, 1, ans_col='wu_day_price')

    merged_data = d.get_data('2014-01-01', end)
    data_drop, _ = preprocess(merged_data, 1, 1, 7, 1, ans_col='wu_day_price')

    retain = []
    drop = []
    for i in range(10):
        res_retain = train_and_eval_model('test', 'large', **data_retain, drop_model=True)
        res_drop = train_and_eval_model('test', 'large', **data_drop, drop_model=True)
        retain.append(res_retain)
        drop.append(res_drop)

    print('retain data before 2013:', retain)
    print('drop data before 2013:', drop)

    ## weather_7dl is mis delete by ken...
# }}}

def exp2():
    # which features to use

    d = RowDataHandler()
    for filename, inpu_method in data_common.items():
        d.add(pd.read_csv(f'{path}/{filename}.csv'), inpu_method)
    for filename, inpu_method in data_wu.items():
        d.add(pd.read_csv(f'{path}/wu/{filename}.csv'), inpu_method)

    d.drop_columns(['sea_pa', 'point_temp', 'rain_hour', 'rain_10m', 'rain_60m', 'shine_hour', 'shine_ratio', 'shine_amount', 'visibility', 'A_evap', 'ultra_high', 'cloud'])
    start, end = d.get_start_end_tick()
    merged_data = d.get_data(start, end)
    columns = np.array((d.get_columns()))
    columns = np.delete(columns, np.where((columns == 'date') | (columns == 'wu_day_price')))

    ori_columns = columns.copy()
    base = ['date', 'wu_day_price']
    for _ in range(5):
        res_list = []
        for i in range(len(columns)):
            mask = np.ones(len(columns), dtype=bool)
            mask[i] = False
            data_mask = columns[mask]
            data, _ = preprocess(merged_data[base + list(data_mask)], 1, 1, 7, 1, ans_col='wu_day_price')
            res = train_and_eval_model('test', 'base', **data, drop_model=True)
            res_list.append(res)
        idx = np.argmin(res_list)
        columns = np.delete(columns, idx)
    print('The origin columns: ')
    print(ori_columns)
    print('The final columns: ')
    print(columns)

if '__main__' == __name__:

    path = './data/use'

    # value is None for daily data
    data_common = {
        'currency': None,
        'powder_feed': ['same', 'same'],
        'yellow_bean': ['same'],
        'weather': None
    }
    data_wu = {
        'wu_export': ['divide', 'same'],
        'wu_price_perDate': None,
        # 'wu_price_perMonth': ['same', 'divide'],
    }
    data_chi = {
        'chi_export': ['divide', 'same'],
        'chi_price_perDate': None,
        'chi_price_perMonth': ['same', 'divede'],
        'chi_small_fish': ['divide', 'same']
    }

    if argv[1] == '1':
        exp1()
    elif argv[1] == '2':
        exp2()
    # main()

