#!/usr/bin/env python3
import numpy as np
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from sys import argv

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.45
# set_session(tf.Session(config = config))
from utils import rmse

batch_size = 128
epochs = 100

def build_class_model(input_shape):

    inputs = Input(shape=input_shape)

    model = Conv1D(32, kernel_size=1, padding='valid', activation='selu', kernel_initializer='lecun_normal')(inputs)
    model = Conv1D(32, kernel_size=1, padding='valid', activation='selu', kernel_initializer='lecun_normal')(model)
    model = Conv1D(64, kernel_size=1, padding='valid', activation='selu', kernel_initializer='lecun_normal')(model)
    model = Conv1D(128, kernel_size=1, padding='valid', activation='selu', kernel_initializer='lecun_normal')(model)
    model = Conv1D(128, kernel_size=input_shape[0], padding='valid', activation='selu', kernel_initializer='lecun_normal')(model)

    model = Flatten()(model)

    model = Dense(128, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(256, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.3)(model)
    model = Dense(512, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.4)(model)
    model = Dense(256, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(128, activation='selu', kernel_initializer='lecun_normal')(model)

    model = Dense(1, activation='selu')(model)

    model = Model(inputs, model)
    model.summary()

    return model

def build_naiive_model(input_shape):
    inputs = Input(shape=input_shape)

    model = Dense(64, activation='selu', kernel_initializer='lecun_normal')(inputs)
    model = Dense(64, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.2)(model)

    model = Dense(128, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(128, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.3)(model)

    model = Dense(256, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(256, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.4)(model)

    model = Dense(512, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.4)(model)

    model = Dense(1024, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.4)(model)

    model = Dense(512, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dropout(0.4)(model)
    model = Dense(256, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(128, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(64, activation='selu', kernel_initializer='lecun_normal')(model)
    model = Dense(1, activation='selu')(model)

    model = Model(inputs, model)
    model.summary()

    return model

def lr_schedule(epoch):
    if epoch < 0.3 * epochs:
        return 0.001
    elif epoch < 0.6 * epochs:
        return 0.0001
    else:
        return 0.00001

if __name__ == '__main__':
    if len(argv) < 3:
        print('usage: ./train.py [mode] [model_name]')
        exit()

    mode=int(argv[1])
    model_name=argv[2]
    assert 1==mode or 2==mode, 'mode must be 1 or 2'
    print('loading data...')
    x, y = np.load(f'tmp/tr_x_{mode}.npy', allow_pickle=True), np.load(f'tmp/tr_y_{mode}.npy', allow_pickle=True)
    print(x.shape)

    model_ckpt = ModelCheckpoint(f'models/model{model_name}.h5', verbose = 1, save_best_only = True)
    tensorboard = TensorBoard(log_dir=f'logs_with_datetime/model{model_name}', histogram_freq=0, write_graph=True, write_images=False)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    print('building model...')
    model = build_naiive_model(x.shape[-1:]) if 1==mode else build_class_model((x.shape[-2], x.shape[-1]))
    model.compile(loss = 'mean_squared_error', optimizer = Adam(lr = 1e-4), metrics = [rmse, 'mae'])
    model.fit(
            x, y,
            batch_size = batch_size,
            epochs = epochs,
            validation_split = 0.4,
            shuffle = True,
            callbacks = [model_ckpt, tensorboard]#, lr_scheduler]
        )
