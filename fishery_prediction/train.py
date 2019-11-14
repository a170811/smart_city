#!/usr/bin/env python3

from sys import argv

import numpy as np
import pandas as pd

from build_set import RowDataHandler, preprocess
from model import train_and_eval_model, linear_regression

def load_wu(start=None, end=None):

    d = RowDataHandler()
    for filename, inpu_method in data_common.items():
        d.add(pd.read_csv(f'{path}/{filename}.csv'), inpu_method)
    for filename, inpu_method in data_wu.items():
        d.add(pd.read_csv(f'{path}/wu/{filename}.csv'), inpu_method)

    if start is None and end is None:
        start, end = d.get_start_end_tick()

    return d, d.get_data(start, end)

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

def exp2():# {{{
    # which features to use

    d = RowDataHandler()
    for filename, inpu_method in data_common.items():
        d.add(pd.read_csv(f'{path}/{filename}.csv'), inpu_method)
    for filename, inpu_method in data_wu.items():
        d.add(pd.read_csv(f'{path}/wu/{filename}.csv'), inpu_method)

    start, end = d.get_start_end_tick()
    merged_data = d.get_data(start, end)
    columns = np.array((d.get_columns()))
    columns = np.delete(columns, np.where((columns == 'date') | (columns == 'wu_day_price')))

    columns_history = []
    mse_history = []
    ori_columns = columns.copy()
    base = ['date', 'wu_day_price']
    for _ in range(len(columns) - 2):
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
        columns_history.append(columns)
        mse_history.append(res_list)
    print('The origin columns: ')
    print(ori_columns)
    print('The final columns: ')
    print(columns)
    for i in range(len(columns_history)):
        print('res: ', mse_history[i])
        print('columns: ', columns_history[i])
        print('')
# }}}


def exp3():# {{{
    # testing for results of exp2, results of selection

    d, data = load_wu()
    use_cols = ['date', 'wu_day_price'] + ['CAD', 'SAR', 'AED', 'pf_weight', 'yb_price', 'pa', 'pa_high', 'pa_low',
      'temp_high', 'temp_low', 'humidity', 'humidity_low', 'wind_dir', 'wind_max',
      'wind_max_dir', 'rainfall', 'wu_ex_price', 'wu_day_amount']

    ori_data, _ = preprocess(data, 1, 1, 7, 1, ans_col='wu_day_price')
    drop_data, _ = preprocess(data[use_cols], 1, 1, 7, 1, ans_col='wu_day_price')
    base_ori = train_and_eval_model('test', 'base', **ori_data, drop_model=True)
    base_select = train_and_eval_model('test', 'base', **drop_data, drop_model=True)
    large_ori = train_and_eval_model('test', 'large', **ori_data, drop_model=True)
    large_select = train_and_eval_model('test', 'large', **drop_data, drop_model=True)

    print('base model: ')
    print('origin columns: ', base_ori)
    print('selection columns: ', base_select)
    print('large model: ')
    print('origin columns: ', large_ori)
    print('selection columns: ', large_select)
# }}}

if '__main__' == __name__:

    # path = './data/use'
    path = './backend'

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

    if 0 == len(argv):
        test()

    if argv[1] == '1':
        exp1()
    elif argv[1] == '2':
        exp2()
    elif argv[1] == '3':
        exp3()

