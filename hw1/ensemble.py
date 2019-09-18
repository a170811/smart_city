#!/usr/bin/env python3
import numpy as np
from keras.models import load_model
import tensorflow as tf

from utils import load_data, rmse

model_name_list = ['hw1_test0.h5', 'hw1_test1.h5']

if '__main__' == __name__:
    dependencies = {
        'rmse': rmse
    }


    test_x, test_y = load_data('features/test.npy')

    models = [load_model(f'models/{name}', custom_objects=dependencies) for name in model_name_list]

    pred_y = 0
    for model in models:
        pred_y = np.add(pred_y, np.squeeze(model.predict(test_x)))
    pred_y = pred_y / len(models)

    print(pred_y)

    sess = tf.Session()
    print(sess.run(rmse(test_y, pred_y)))

