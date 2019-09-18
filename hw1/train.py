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

def build_model(feature_length):
    inputs = Input(shape=(feature_length,))

    model = Dense(64, activation='selu', kernel_initializer='lecun_normal')(inputs)
    model = Dense(64, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(64, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.2)(model)

    model = Dense(128, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(128, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(128, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.3)(model)

    model = Dense(256, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(256, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(256, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.4)(model)

    model = Dense(512, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(512, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.4)(model)

    model = Dense(1024, activation='selu', kernel_initializer='lecun_normal')(model)
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
    batch_size = 16
    epochs = 100

    x, y = np.load('tmp/tr_x.npy'), np.load('tmp/tr_y.npy')
    print(x.shape, y.shape)
    feature_length = x.shape[-1]

    model_ckpt = ModelCheckpoint(f'models/{model_name}.h5', verbose = 1, save_best_only = True)
    tensorboard = TensorBoard(log_dir=f'logs/{model_name}', histogram_freq=0, write_graph=True, write_images=False)

    model = build_model(feature_length)
    model.compile(loss = 'mean_squared_error', optimizer = Adam(lr = 1e-3), metrics = [rmse])
    model.fit(
            x, y,
            batch_size = batch_size,
            epochs = epochs,
            validation_split = 0.4,
            shuffle = True,
            callbacks = [model_ckpt, tensorboard]
        )
