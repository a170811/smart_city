#!/usr/bin/env python3

from keras.models import load_model
import numpy as np
import pandas as pd

from sys import argv

# local import
from train import load_data
from train import data_common, data_wu, data_chi, path
from build_set import preprocess, split


def predict_res(model_name, data):

    model = load_model(model_name)
    res = model.predict(data)
    return res

def to_dataframe(model_name, x, y, t):
    data_len = len(x)
    res = predict_res(model_name, x)

    y = y.reshape(data_len, -1)
    res = res.reshape(data_len, -1)

    ans = y.reshape(data_len, -1)
    pred = res.reshape(data_len, -1)
    dict_res = {'time': t}
    for i in range(ans.shape[-1]):
        dict_res[f'pred{i+1}'] = pred[:, i]
        dict_res[f'ans{i+1}'] = ans[:, i]
    res = pd.DataFrame(dict_res)

    return res

if '__main__' == __name__:

    if len(argv) < 4:
        print(f'Useage: python3 {argv[0]} [wu/chi] [day/week] [model name]')
        exit()

    target = argv[1]
    mode = argv[2]
    model_name = f'./models/{argv[3]}'
    data, time = load_data(target, mode)

    tr = to_dataframe(model_name, data['train_x'], data['train_y'], time['train_t'])
    va = to_dataframe(model_name, data['valid_x'], data['valid_y'], time['valid_t'])
    te = to_dataframe(model_name, data['test_x'], data['test_y'], time['test_t'])
    res = pd.concat([tr, va, te], ignore_index=True)
    print(res)
    filename = f'{target}_{mode}'
    res.to_csv(f'{filename}.csv', index=False)
    with open(f'{filename}.info', 'w') as f:
        f.write(f"train: {tr['time'].iloc[0].__format__('%Y-%m-%d')} to {tr['time'].iloc[-1].__format__('%Y-%m-%d')}\n")
        f.write(f"valid: {va['time'].iloc[0].__format__('%Y-%m-%d')} to {va['time'].iloc[-1].__format__('%Y-%m-%d')}\n")
        f.write(f"test: {te['time'].iloc[0].__format__('%Y-%m-%d')} to {te['time'].iloc[-1].__format__('%Y-%m-%d')}\n")
