#!/usr/bin/env python3
import numpy as np
from keras import backend as K

# def load_data(file_name):
#     feature = np.load(file_name)
#
#     x, y = [], []
#
#     for data in feature:
#         x.append(data.get('x'))
#         y.append(data.get('y'))
#
#     x = np.array(x).astype(float)
#     y = np.array(y).astype(float)
#
#     return x, y

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
