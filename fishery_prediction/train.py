#!/usr/bin/env python3

import pandas as pd

from data_merger import RowDataHandler
from model import train_and_eval_model, linear_regression

if '__main__' == __name__:


    path = './data/use'

    # value is None for daily data
    data_common = {
        'currency': None,
        'powder_feed': ['same', 'same'],
        'yellow_bean': ['same'],
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

    # data = d.preprocess(d.get_columns(), 1, 7, 1)
    data = d.preprocess(['date', 'wu_day_price'], 1, 7, 1)
    train_and_eval_model('deep_dropout', 'base', **data)

    linear_regression(data['test_x'], data['test_y'])
