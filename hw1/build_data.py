import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def data_format(data, ohe):
    ohe_part = ohe.transform(data[:, :3]).toarray()
    num_part = data[:, 3:] - [22, 120]
    return np.concatenate((ohe_part, num_part), axis=1)

columns = ['Id', 'time', 'district', 'administration', 'type', 'content', 'latitude', 'longitude']
df_train, df_test = pd.read_csv('./data/train.txt', header=None, names=columns+['cost_time']), pd.read_csv('./data/test.txt', header=None, names=columns)
drop = ['Id', 'time', 'content']
df_train = df_train.drop(columns=drop)
df_test = df_test.drop(columns=drop)


## data
tr_x, tr_y = df_train.iloc[:, :-1].to_numpy(), df_train.iloc[:, -1].to_numpy()
to_predict = df_test.to_numpy()
ohe = OneHotEncoder(handle_unknown='ignore').fit(tr_x[:, :3])

tr_x = data_format(tr_x, ohe)
to_predict = data_format(to_predict, ohe)

# print(np.shape(tr_x))
# print(np.shape(to_predict))

tr_x, te_x, tr_y, te_y = train_test_split(tr_x, tr_y, test_size = 0.1)

print(tr_x[:2])
exit()

print('tr_x: ', np.shape(tr_x))
print('tr_y: ', np.shape(tr_y))
print('te_x: ', np.shape(te_x))
print('te_y: ', np.shape(te_y))
print('to_predict: ', np.shape(to_predict))

np.save('./tmp/tr_x.npy', tr_x)
np.save('./tmp/te_x.npy', te_x)
np.save('./tmp/tr_y.npy', tr_y)
np.save('./tmp/te_y.npy', te_y)
np.save('./tmp/to_predict.npy', to_predict)
