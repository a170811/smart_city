#!/usr/bin/env python3

import numpy as np
import os
from sys import argv

from keras.models import Sequential
from keras.layers import Input, LSTM, Lambda, Conv1D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

def build_model(unit_size, num_input_unit, num_output_unit):

    model = Sequential()
    model.add(LSTM(256, activation='relu', input_shape=(num_input_unit, unit_size), return_sequences=True))
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(LSTM(1, return_sequences=True))
    model.add(Lambda(lambda x: x[:, -num_output_unit:, :]))
    model.summary()
    return model

def build_large_model(unit_size, num_input_unit, num_output_unit):

    inputs = Input(shape=(num_input_unit, unit_size))
    model = inputs
    model = LSTM(512, activation='relu', return_sequences=True)(model)
    model = LSTM(512, activation='relu', return_sequences=True)(model)
    model = Dropout(0.3)(model)
    model = LSTM(1024, activation='relu', return_sequences=True)(model)

    model = Conv1D(128, 1, activation='relu')(model)
    model = Conv1D(128, num_input_unit, activation='relu')(model)

    model = LSTM(1, return_sequences=True)(model)
    model = Lambda(lambda x: x[:, -num_output_unit:, :])(model)

    model = Model(inputs, model)
    model.summary()

    return model

if __name__ == '__main__':

    model_name = f'LSTM_{argv[1]}_{argv[2]}'
    if os.path.isfile(f'models/{model_name}.h5'):
        print('model existed')
        exit()

    train_x = np.load('features/train_x.npy')
    train_y = np.load('features/train_y.npy')
    valid_x = np.load('features/valid_x.npy')
    valid_y = np.load('features/valid_y.npy')
    test_x = np.load('features/test_x.npy')
    test_y = np.load('features/test_y.npy')
    print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape, test_x.shape, test_y.shape)

    if not os.path.exists('models'):
            os.makedirs('models')
    model_ckpt = ModelCheckpoint(f'models/{model_name}.h5', verbose = 1, save_best_only = True)
    if not os.path.exists('runs'):
            os.makedirs('runs')
    tensorboard = TensorBoard(log_dir='runs/%s' % model_name , histogram_freq=0, write_graph=True, write_images=False)
    early_stp = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")

    if 'base' == argv[1]:
        model = build_model(train_x.shape[2], train_x.shape[1], train_y.shape[1])
    elif 'large' == argv[1]:
        model = build_large_model(train_x.shape[2], train_x.shape[1], train_y.shape[1])


    model.compile(loss = 'mean_squared_error', optimizer = Adam(lr = 1e-3))
    model.fit(train_x, train_y,\
	validation_data=(valid_x, valid_y),\
	shuffle=True,\
	batch_size=32,\
	epochs=1000,\
	callbacks=[model_ckpt, tensorboard, early_stp])
    mse = model.evaluate(test_x, test_y)
    print(mse)
