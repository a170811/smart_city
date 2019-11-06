#!/usr/bin/env python3

import numpy as np
from sklearn.preprocessing import normalize
from sys import argv
import pandas as pd

def preprocess(df, columns, unit_length=7, num_input_unit=4, num_output_unit=1):
    df = df[columns]
    df = df.dropna()

    dates = df['date']
    content = df[columns[1:]]

    units = []
    for i in range(unit_length):
        unit = build_unit(dates, content, i).set_index('date')
        units.append(unit)

    units = pd.concat(units).groupby('date').mean()
    units = units[unit_length - 1:]

    train = units[~(units.index.str.contains('2018')+units.index.str.contains('2019'))]
    valid = units[units.index.str.contains('2018')]
    test = units[units.index.str.contains('2019')]

    train_x, train_y = serialize(train, num_input_unit, num_output_unit)
    valid_x, valid_y = serialize(valid, num_input_unit, num_output_unit)
    test_x, test_y = serialize(test, num_input_unit, num_output_unit)

    np.save('features/train_x.npy', train_x)
    np.save('features/train_y.npy', train_y)
    np.save('features/valid_x.npy', valid_x)
    np.save('features/valid_y.npy', valid_y)
    np.save('features/test_x.npy', test_x)
    np.save('features/test_y.npy', test_y)

def build_unit(dates, content, i_unit):
    dates = dates[i_unit:].reset_index(drop=True)
    content = content[:len(dates)].reset_index(drop=True)
    content['amount(kg)'] = content['amount(kg)'].str.replace(',', '')
    content = content.astype(float)

    return pd.concat([dates, content], axis=1, sort=False, ignore_index=False)

def serialize(dataset, num_input_unit, num_output_unit):

    dataset = normalize(dataset)
    feature_length = dataset.shape[0] - num_input_unit -num_output_unit + 1
    feature_x = np.zeros([feature_length, num_input_unit, dataset.shape[1]])
    for i in range(feature_length):
        for j in range(num_input_unit):
            for k in range(dataset.shape[1]):
                feature_x[i, j, k] = dataset[i + j, k]

    feature_y = np.zeros([feature_length, num_output_unit, 1])
    for i in range(feature_length):
        for j in range(num_output_unit):
            feature_y[i, j, 0] = dataset[i + j + num_input_unit, -1]

    return feature_x, feature_y

if '__main__' == __name__:

    if len(argv) != 4:
        print('Usage: ./preprocess.py unit_length, num_input_unit, num_output_unit')
        exit()

    unit_length = int(argv[1])
    num_input_unit = int(argv[2])
    num_output_unit = int(argv[3])

    df = pd.read_csv('data/chi.csv')
    preprocess(df, ['date', '最低氣溫(℃)', 'amount(kg)', 'mean price(doller/kg)'],\
               unit_length, num_input_unit, num_output_unit)
