from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sys import argv


def data_format1(tr_x, to_predict):
    ohe = OneHotEncoder(handle_unknown='ignore').fit(tr_x[:, :-2])
    tr_ohe = ohe.transform(tr_x[:, :-2]).toarray()
    tr_num = tr_x[:, -2:] - [22, 120]
    predict_ohe = ohe.transform(to_predict[:, :-2]).toarray()
    predict_num = to_predict[:, -2:] - [22, 120]
    return np.concatenate((tr_ohe, tr_num), axis=1), np.concatenate((predict_ohe, predict_num), axis=1)

def data_format2(tr_x, to_predict):
    ohe = OneHotEncoder(handle_unknown='ignore').fit(tr_x[:, :3])
    tr_ohe_ = ohe.transform(tr_x[:, :3]).toarray()
    predict_ohe_ = ohe.transform(to_predict[:, :3]).toarray()

    cat_idx = [len(i) for i in ohe.categories_]
    fea_len = max(cat_idx)
    n_tr, n_predict = len(tr_x), len(to_predict)

    tr_ohe = np.zeros((n_tr, len(cat_idx)+2, fea_len))
    predict_ohe = np.zeros((n_predict, len(cat_idx)+2, fea_len))
    s = 0
    e = 0
    for i, l in enumerate(cat_idx):
        e += l
        tr_ohe[:, i, :l] = tr_ohe_[:, s:e]
        tr_ohe[:, 3:, 0] = tr_x[:, 3:] - [22, 120]
        predict_ohe[:, i, :l] = predict_ohe_[:, s:e]
        predict_ohe[:, 3:, 0] = to_predict[:, 3:] - [22, 120]
        s += l

    return tr_ohe, predict_ohe

def hour_classify(time, num_class):
    return time.hour // num_class

if '__main__' == __name__:

    if len(argv) < 2:
        print('Useage: python3 build_data [format]')
        exit()
    format_type = int(argv[1])

    columns = ['Id', 'time', 'district', 'administration', 'type', 'content', 'latitude', 'longitude']
    df_train, df_test = pd.read_csv('./data/train.txt', header=None, names=columns+['cost_time']), pd.read_csv('./data/test.txt', header=None, names=columns)

    df_train['time'] = [datetime.strptime(time, "%Y-%m-%d %H:%M:%S") for time in df_train['time']]
    df_test['time'] = [datetime.strptime(time, "%Y-%m-%d %H:%M:%S") for time in df_test['time']]

    df_train['weekday'] = [time.weekday() for time in df_train['time']]
    df_test['weekday'] = [time.weekday() for time in df_test['time']]

    df_train['8hrs'] = [hour_classify(time, 8) for time in df_train['time']]
    df_test['8hrs'] = [hour_classify(time, 8) for time in df_test['time']]

    cols_train = ['district', 'administration', 'type', 'weekday', '8hrs', 'latitude', 'longitude', 'cost_time']
    df_train = df_train[cols_train]
    cols_test = ['district', 'administration', 'type', 'weekday', '8hrs', 'latitude', 'longitude']
    df_test = df_test[cols_test]

    # data
    tr_x, tr_y = df_train.iloc[:, :-1].to_numpy(), df_train.iloc[:, -1].to_numpy()
    to_predict = df_test.to_numpy()

    if 1 == format_type:
        tr_x, to_predict = data_format1(tr_x, to_predict)
    # elif 2 == format_type:
    #     tr_x, to_predict = data_format2(tr_x, to_predict)
    else:
        print(f'no format_type: {format_tyupe}')
        exit()

    print(tr_x, to_predict)
    tr_x, te_x, tr_y, te_y = train_test_split(tr_x, tr_y, test_size = 0.1)

    print('tr_x: ', np.shape(tr_x))
    print('tr_y: ', np.shape(tr_y))
    print('te_x: ', np.shape(te_x))
    print('te_y: ', np.shape(te_y))
    print('to_predict: ', np.shape(to_predict))

    np.save(f'./tmp/tr_x_{format_type}.npy', tr_x)
    np.save(f'./tmp/te_x_{format_type}.npy', te_x)
    np.save(f'./tmp/tr_y_{format_type}.npy', tr_y)
    np.save(f'./tmp/te_y_{format_type}.npy', te_y)
    np.save(f'./tmp/to_predict_{format_type}.npy', to_predict)
