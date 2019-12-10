#!/usr/bin/env python3

from calendar import monthrange
from datetime import datetime
import numpy as np
import pandas as pd
import pickle


class RowDataHandler():# {{{

    def __init__(self):

        self.df = pd.DataFrame(index=pd.date_range('1990-01-01', '2020-12-31'))

    def add(self, df_path, inputations):
        # df(dataframe): first col must be 'date'
        # inputation: monthly data will be inputated to daily data
        #             `divide`, `same`

        df_load = pd.read_csv(df_path, index_col='date')

        # drop na columns
        sparse_columns = df_load.columns[df_load.isnull().mean() > 0.5].to_list()
        if 0 != len(sparse_columns):
            print('\ndrop sparse col: ')
            print(sparse_columns)
            print('')
        df_load = df_load.drop(columns=sparse_columns)

        df_load = df_load.astype(str)
        df_load.index = pd.to_datetime(df_load.index)
        df_load = df_load.apply(lambda x: x.str.replace(',', ''))
        df_load = df_load.astype(float)

        if inputations is not None:
            days_list = df_load.index.map(lambda x: monthrange(x.year, x.month)[1])
            for col, inputation in zip(df_load.columns, inputations):
                if 'divide' == inputation:
                    df_load[col] /= days_list
            df_new = pd.DataFrame(np.repeat(df_load.to_numpy(), days_list,\
                        axis=0), columns=df_load.columns)
            df_new.index = pd.date_range(df_load.index[0], periods=np.sum(days_list))
        else:
            df_new = df_load

        self.df = pd.merge(self.df, df_new, left_index=True, right_index=True)

    def drop_columns(self, columns):
        self.df = self.df.drop(columns=columns)

    def get_merged_data(self, start, end):

        assert len(self.df) > 0, 'there is no data in handler, please add first!!'
        start = datetime.strptime(start, '%Y-%m-%d')
        end = datetime.strptime(end, '%Y-%m-%d')
        df_start = self.df.index[0]
        df_end = self.df.index[-1]
        assert (start >= df_start) & (end <= df_end), 'timestamp is out of data'

        res = self.df.loc[(self.df.index >= start) & (self.df.index <= end)]
        return res

    def get_columns(self):
        return self.df.columns

    def get_start_end_tick(self):
        date = self.df.index
        start = date[0].__format__('%Y-%m-%d')
        end = date[-1].__format__('%Y-%m-%d')
        return start, end
# }}}


def preprocess(data, ans, date, unit_length=7, sample_stride=7, num_used_unit=4,# {{{
                num_pred_unit=1):
    #       unit length
    #     |<---->| (mean)
    # ------------------------------- time ------------------------------->
    #            ^        ^        ^
    #            |        |        | sample stride
    #            t        t        t
    # ** the timestemp is refer to the first day of each unit

    # future
    #
    #            1 ... num_used_unit ...  N             1 .. num_pred_unit .. N
    #        use unit                 pred unit    pred unit              pred unit
    #     |<--(mean)-->|    ...    |<--(mean)-->|<--(mean)-->|    ...   |<--(mean)-->|
    # ------------------------------- time ------------------------------------------->
    #                                           ^        ^        ^ (N used_unit)
    #                                           |        |        | sample stride
    #                                           t        t        t
    # ** the timestemp is refer to the first day of each unit

    assert len(data) == len(ans), 'length of data and ans are mismatch!!'
    data = data.reset_index(drop=True)
    data = data.fillna(0)

    unit_x, unit_y, unit_t = [], [], []
    for i in range(0, len(data), sample_stride):
        end = i + unit_length
        if end > len(data):
            break
        x = data.iloc[i: end, :].mean(axis=0).to_list()
        y = ans.iloc[i: end].mean(axis=0).to_list()
        t = date[end - 1]
        # t = date[i]
        unit_x.append(x)
        unit_y.append(y)
        unit_t.append(t)

    end = num_used_unit + num_pred_unit
    set_x, set_y, set_t = [], [], []
    for i in range(len(unit_x)):
        if (i + end) > len(unit_x):
            break
        x = unit_x[i:i+num_used_unit]
        y = unit_y[i+num_used_unit:i+end]
        set_x.append(x)
        set_y.append(y)
        set_t.append(unit_t[i+num_used_unit-1])
    set_x, set_y, set_t = np.array(set_x), np.array(set_y), np.array(set_t)

    return (set_x, set_y, set_t)

# }}}


def split(data, tr_ratio, va_ratio, te_ratio = 0):

    assert 1 == tr_ratio + va_ratio + te_ratio, 'split ratio error'

    set_x, set_y, set_t = data
    s1 = tr_ratio
    s2 = tr_ratio + va_ratio
    split_point = [int(len(set_x)*s1), int(len(set_x)*s2)]
    tr_x, va_x, te_x = np.split(set_x, split_point)
    tr_y, va_y, te_y = np.split(set_y, split_point)
    tr_t, va_t, te_t = np.split(set_t, split_point)

    data_res = {
        'train_x': tr_x,
        'train_y': tr_y,
        'valid_x': va_x,
        'valid_y': va_y,
        'test_x': te_x,
        'test_y': te_y,
    }
    time_res = {
        'train_t': tr_t,
        'valid_t': va_t,
        'test_t': te_t,
    }
    return data_res, time_res

if '__main__' == __name__:

    path = './data/use'

    # value is None for daily data
    data_common = {
        'currency': None,
        'powder_feed': ['same', 'same'],
        'yellow_bean': ['same'],
        'weather': None,
    }
    data_wu = {
        'wu_export': ['divide', 'same'],
        'wu_price_perDate': None,
        'wu_price_perMonth': ['same', 'divide'],
    }
    data_chi = {
        'chi_export': ['divide', 'same'],
        'chi_price_perDate': None,
        'chi_price_perMonth': ['same', 'divede'],
        'chi_small_fish': ['divide', 'same']
    }

    d = RowDataHandler()
    # d.add(pd.read_csv(f'{path}/powder_feed.csv'), 'same')
    # d.add(pd.read_csv(f'{path}/wu/wu_price_perMonth.csv'), 'same')
    for filename, inpu_method in data_common.items():
        d.add(f'{path}/{filename}.csv', inpu_method)
    for filename, inpu_method in data_wu.items():
        d.add(f'{path}/wu/{filename}.csv', inpu_method)

    start, end = d.get_start_end_tick()
    merged_data = d.get_merged_data(start, end)

    # col = d.get_columns()[5:]
    # d.drop_columns(col)
    # res = d.get_merged_data(*d.get_start_end_tick())

    data, ans , date = merged_data, merged_data[['wu_day_price']],\
                                merged_data.index

    columns = data.columns
    processed_data = preprocess(data, ans, date)

    data, time = split(processed_data, tr_ratio=0.8, va_ratio=0.1, te_ratio=0.1)
    pickle.dump(data, open('./data/wu_data.pkl', 'wb'))
    pickle.dump(time, open('./data/wu_time.pkl', 'wb'))
    pickle.dump(columns, open('./data/wu_columns.pkl', 'wb'))
