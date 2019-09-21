from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sys import argv


def data_format1(tr_x, to_predict):
    ohe = OneHotEncoder(handle_unknown='ignore').fit(tr_x[:, :-2])
    tr_ohe = ohe.transform(tr_x[:, :-2]).toarray()
    tr_num = tr_x[:, -2:]
    predict_ohe = ohe.transform(to_predict[:, :-2]).toarray()
    predict_num = to_predict[:, -2:]

    return np.concatenate((tr_ohe, tr_num), axis=1), np.concatenate((predict_ohe, predict_num), axis=1)

def data_format2(tr_x, to_predict):
    ohe = OneHotEncoder(handle_unknown='ignore').fit(tr_x[:, :-2])
    category_dic = ohe.categories_

    categories_len = [len(i) for i in category_dic]
    feature_len = max(categories_len)

    tr_ohe = []
    for event in tr_x:
        event_ohe = np.zeros((len(categories_len)+2, feature_len))
        for i, feature in enumerate(event):
            if  i < tr_x.shape[1] - 2:
                idx = np.where(category_dic[i] == feature)[0]
                event_ohe[i][idx] = 1. if idx != -1 else 0.
            else:
                event_ohe[i][0] = feature
        tr_ohe.append(event_ohe)

    predict_ohe = []
    for event in to_predict:
        event_ohe = np.zeros((len(categories_len)+2, feature_len))
        for i, feature in enumerate(event):
            if  i < tr_x.shape[1] - 2:
                idx = np.where(category_dic[i] == feature)[0]
                event_ohe[i][idx] = 1. if idx != -1 else 0.
            else:
                event_ohe[i][0] = feature
        predict_ohe.append(event_ohe)

    return np.array(tr_ohe), np.array(predict_ohe)

def hour_classify(time, num_class):
    return time.hour // num_class

if '__main__' == __name__:

    if len(argv) < 2:
        print('Useage: python3 build_data [format]')
        exit()
    format_type = int(argv[1])

    columns = ['Id', 'time', 'district', 'administration', 'type', 'content', 'latitude', 'longitude']
    df_train, df_test = pd.read_csv('./data/train.txt', header=None, names=columns+['cost_time']), pd.read_csv('./data/test.txt', header=None, names=columns)

    df_train['time'] = df_train['time'].astype('datetime64[ns]')
    df_test['time'] = df_test['time'].astype('datetime64[ns]')

    df_train['weekday'] = [time.weekday() for time in df_train['time']]
    df_test['weekday'] = [time.weekday() for time in df_test['time']]

    df_train['hour'] = [hour_classify(time, 1) for time in df_train['time']]
    df_test['hour'] = [hour_classify(time, 1) for time in df_test['time']]

    cols_train = ['district', 'administration', 'type', 'weekday', 'hour', 'latitude', 'longitude', 'cost_time']
    df_train = df_train[cols_train]
    df_train['latitude'] = df_train['latitude'] - 22
    df_train['longitude'] = df_train['longitude'] - 120
    cols_test = ['district', 'administration', 'type', 'weekday', 'hour', 'latitude', 'longitude']
    df_test = df_test[cols_test]
    df_test['latitude'] = df_test['latitude'] - 22
    df_test['longitude'] = df_test['longitude'] - 120

    # data
    tr_x, tr_y = df_train.iloc[:, :-1].to_numpy(), df_train.iloc[:, -1].to_numpy()
    to_predict = df_test.to_numpy()

    if 1 == format_type:
        tr_x, to_predict = data_format1(tr_x, to_predict)
    elif 2 == format_type:
        tr_x, to_predict = data_format2(tr_x, to_predict)
    else:
        print(f'no format_type: {format_type}')
        exit()

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
