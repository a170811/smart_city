#!/usr/bin/env python3
import numpy as np
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from sys import argv

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.45
# set_session(tf.Session(config = config))
from utils import rmse

def build_class_model(input_shape):
    inputs = Input(shape=input_shape)

    model = Conv1D(32, kernel_size=1, padding='valid', activation='selu', kernel_initializer='lecun_normal')(inputs)
    model = Conv1D(64, kernel_size=1, padding='valid', activation='selu', kernel_initializer='lecun_normal')(model)
    model = Conv1D(128, kernel_size=input_shape[0], padding='valid', activation='selu', kernel_initializer='lecun_normal')(model)

    model = Flatten()(model)

    model = Dense(128, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(128, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(256, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(256, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(512, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.4)(model)

    model = Dense(1, activation='selu')(model)

    model = Model(inputs, model)
    model.summary()

    return model

def build_naiive_model(input_shape):
    model = Input(shape=input_shape)

    model = Dense(64, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(64, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.2)(model)

    model = Dense(128, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(128, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.3)(model)

    model = Dense(256, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(256, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.4)(model)

    model = Dense(512, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(512, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.4)(model)

    model = Dense(1, activation='selu')(model)

    model = Model(inputs, model)
    model.summary()

    return model

if __name__ == '__main__':
    if len(argv) < 2:
        print('usage: ./train.py model_name')
        exit()

    model_name=argv[1]
    batch_size = 64
    epochs = 100

    print('loading data...')
    x, y = np.load('tmp/tr_x_2.npy'), np.load('tmp/tr_y_2.npy')

    model_ckpt = ModelCheckpoint(f'models/{model_name}.h5', verbose = 1, save_best_only = True)
    tensorboard = TensorBoard(log_dir=f'logs/{model_name}', histogram_freq=0, write_graph=True, write_images=False)

    print('building model...')
    # model = build_naiive_model(x.shape[-1])
    model = build_class_model((x.shape[-2], x.shape[-1]))
    model.compile(loss = 'mean_squared_error', optimizer = Adam(lr = 1e-3), metrics = [rmse, 'mae'])
    model.fit(
            x, y,
            batch_size = batch_size,
            epochs = epochs,
            validation_split = 0.4,
            shuffle = True,
            callbacks = [model_ckpt, tensorboard]
        )
