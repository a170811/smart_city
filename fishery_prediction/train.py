#!/usr/bin/env python3

from sys import argv

import numpy as np
import pandas as pd

from build_set import RowDataHandler, preprocess
from model import train_and_eval_model, linear_regression

def load_wu(columns=None, start=None, end=None):

    d = RowDataHandler()
    for filename, inpu_method in data_common.items():
        d.add(f'{path}/{filename}.csv', inpu_method)
    for filename, inpu_method in data_wu.items():
        d.add(f'{path}/wu/{filename}.csv', inpu_method)

    if start is None and end is None:
        start, end = d.get_start_end_tick()
    merged_data = d.get_merged_data(start, end)
    input_data, ans, date = merged_data, merged_data[['wu_day_price']],\
                      merged_data.index
    if columns is not None:
        input_data = input_data[columns]

    return input_data, ans, date

def load_chi(columns=None, start=None, end=None):

    d = RowDataHandler()
    for filename, inpu_method in data_common.items():
        d.add(f'{path}/{filename}.csv', inpu_method)
    for filename, inpu_method in data_chi.items():
        d.add(f'{path}/chi/{filename}.csv', inpu_method)

    if start is None and end is None:
        start, end = d.get_start_end_tick()
    merged_data = d.get_merged_data(start, end)
    input_data, ans, date = merged_data, merged_data[['chi_day_price']],\
                      merged_data.index
    if columns is not None:
        input_data = input_data[columns]

    return input_data, ans, date

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
    merged_data = d.get_merged_data(*d.get_start_end_tick())[['date', 'wu_day_price']]
    data, time = preprocess(merged_data, 1, 1, 7, 1)
    linear_regression(data['test_x'], data['test_y'])

    merged_data = d.get_merged_data(*d.get_start_end_tick())
    merged_data = merged_data[['date', 'temp', 'temp_high', 'temp_low', 'wu_ex_weight', 'wu_ex_price', 'wu_day_price', 'wu_day_amount']]
    merged_data = merged_data.dropna().reset_index(drop=True)
    data, time = preprocess(merged_data, 1, 1, 7, 1)
    train_and_eval_model('weather_7dl', 'large', **data)

    merged_data = d.get_merged_data(*d.get_start_end_tick())
    merged_data = merged_data[['date', 'temp', 'temp_high', 'temp_low', 'wu_ex_weight', 'wu_ex_price', 'wu_day_price', 'wu_day_amount']]
    merged_data = merged_data.dropna().reset_index(drop=True)
    data, time = preprocess(merged_data, 1, 1, 14, 1)
    train_and_eval_model('weather_14dl', 'large', **data)

    merged_data = d.get_merged_data(*d.get_start_end_tick())
    merged_data = merged_data[['date', 'temp', 'temp_high', 'temp_low', 'wu_ex_weight', 'wu_ex_price', 'wu_day_price', 'wu_day_amount']]
    merged_data = merged_data.dropna().reset_index(drop=True)
    data, time = preprocess(merged_data, 7, 7, 4, 1)
    train_and_eval_model('weather_4wl', 'large', **data)

    merged_data = d.get_merged_data(*d.get_start_end_tick())
    merged_data = merged_data[['date', 'temp', 'temp_high', 'temp_low', 'wu_ex_weight', 'wu_ex_price', 'wu_day_price', 'wu_day_amount']]
    merged_data = merged_data.dropna().reset_index(drop=True)
    data, time = preprocess(merged_data, 30, 30, 4, 1)
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
    merged_data = d.get_merged_data(start, end)
    data, ans, date = merged_data.iloc[:, 1:], merged_data['wu_day_price'],\
                      merged_data['date']
    data_retain, _ = preprocess(data, ans, date, 1, 1, 7, 1)

    merged_data = d.get_merged_data('2014-01-01', end)
    data_drop, _ = preprocess(merged_data, 1, 1, 7, 1)

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

    input_data, ans, date = load_wu()
    columns = input_data.columns.to_numpy()

    columns_history = []
    mse_history = []
    ori_columns = columns.copy()
    for _ in range(len(columns) - 2):
        res_list = []
        for i in range(len(columns)):
            mask = np.ones(len(columns), dtype=bool)
            mask[i] = False
            data_mask = columns[mask]
            data, _ = preprocess(input_data[data_mask], ans, date, 1, 1, 7, 1)
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

    target = 'wu'

    if 'wu' == target:
        input_data, ans, date = load_wu()
    elif 'chi' == target:
        input_data, ans, date = load_chi()
    else:
        raise Exception(f'Error target: {target}')

    with open(f'logs/selec_res_{target}', 'r') as f:
        cols_set = [line.strip()[11:-1].replace('\'', '').split(' ') for line in f.readlines()]

    for use_cols in cols_set:
        input_data = input_data[use_cols]
        data, _ = preprocess(input_data, ans, date, 1, 1, 7, 1)
        res_base = []
        res_large = []

        for i in range(1):
            res1 = train_and_eval_model('test', 'base', **data, drop_model=True)
            res2 = train_and_eval_model('test', 'large', **data, drop_model=True)
            res_base.append(res1)
            res_large.append(res2)

        with open(f'logs/exp3_eval_selection_{target}', 'a+') as f:
            f.write(f'1.\n')
            f.write(f'Use columns: {use_cols}\n')
            f.write(f'base model: {res_base}\n')
            f.write(f'base mean: {np.mean(res_base)}\n')
            f.write(f'large model: {res_large}\n')
            f.write(f'large mean: {np.mean(res_large)}\n\n')

        break

# }}}

def exp4():# {{{

    input_data, ans, date = load_wu(['SAR', 'yb_price', 'pa', 'humidity_low', 'wind_max_dir', 'wu_day_price', 'wu_day_amount'])

    for input_size in range(1, 15):
        mses=[]
        for i in range(5):
            data, _ = preprocess(input_data, ans, date, 1, 1, input_size, 3)

            npad = ((0, 0), (0, 3), (0, 0))
            data['train_x'] = np.pad(data['train_x'], npad, 'constant', constant_values=0)
            data['valid_x'] = np.pad(data['valid_x'], npad, 'constant', constant_values=0)
            data['test_x'] = np.pad(data['test_x'], npad, 'constant', constant_values=0)

            mse = train_and_eval_model('test', 'base', **data, drop_model=True)
            mses.append(mse)
        print(f'inupt_size={input_size}\nmse={np.mean(mse)}')
# }}}

def exp5():# {{{

    input_data, ans, date = load_wu(['SAR', 'yb_price', 'pa', 'humidity_low', 'wind_max_dir', 'wu_day_price', 'wu_day_amount'])

    for input_size in range(1, 15):
        mses=[]
        for i in range(5):
            data, _ = preprocess(input_data, ans, date, 7, 7, input_size, 2)

            npad = ((0, 0), (0, 3), (0, 0))
            data['train_x'] = np.pad(data['train_x'], npad, 'constant', constant_values=0)
            data['valid_x'] = np.pad(data['valid_x'], npad, 'constant', constant_values=0)
            data['test_x'] = np.pad(data['test_x'], npad, 'constant', constant_values=0)

            mse = train_and_eval_model('test', 'base', **data, drop_model=True)
            mses.append(mse)
        print(f'inupt_size={input_size}\nmse={np.mean(mse)}')
# }}}

def exp6():# {{{

    input_data, ans, date = load_wu(['SAR', 'yb_price', 'pa', 'humidity_low', 'wind_max_dir', 'wu_day_price', 'wu_day_amount'])

    for input_size in range(1, 15):
        mses=[]
        for i in range(5):
            data, _ = preprocess(input_data, ans, date, 30, 30, input_size, 1)

            npad = ((0, 0), (0, 3), (0, 0))
            data['train_x'] = np.pad(data['train_x'], npad, 'constant', constant_values=0)
            data['valid_x'] = np.pad(data['valid_x'], npad, 'constant', constant_values=0)
            data['test_x'] = np.pad(data['test_x'], npad, 'constant', constant_values=0)

            mse = train_and_eval_model('test', 'base', **data, drop_model=True)
            mses.append(mse)
        print(f'inupt_size={input_size}\nmse={np.mean(mse)}')
# }}}

def test():
    input_data, ans, date = load_wu(['SAR', 'yb_price', 'pa', 'humidity_low', 'wind_max_dir', 'wu_day_price', 'wu_day_amount'])
    data, _ = preprocess(input_data, ans, date, 1, 1, 7, 1)
    res = train_and_eval_model('test', 'base', **data, drop_model=True)
    print(res)

if '__main__' == __name__:

    path = './data/use'
    # path = './backend'

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
        'wu_price_perMonth': ['same', 'divide'],
    }
    data_chi = {
        'chi_export': ['divide', 'same'],
        'chi_price_perDate': None,
        'chi_price_perMonth': ['same', 'divede'],
        'chi_small_fish': ['divide', 'same']
    }

    if 1 == len(argv):
        test()
        exit()

    if argv[1] == '1':
        exp1()
    elif argv[1] == '2':
        exp2()
    elif argv[1] == '3':
        exp3()
    elif argv[1] == '4':
        exp4()

