#!/usr/bin/env python3

import pandas as pd

from build_set import RowDataHandler, preprocess
from model import train_and_eval_model, linear_regression

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

