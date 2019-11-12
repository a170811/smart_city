#!/usr/bin/env python3

from calendar import monthrange
from datetime import datetime
import numpy as np
import pandas as pd

class RowDataHandler():
    def __init__(self):

        self.df = pd.DataFrame(pd.date_range('1990-01-01', '2020-12-31'), columns=['date'])

    def add(self, df_load, inputations):
        # df(dataframe): first col must be 'date'
        # inputation: monthly data will be inputated to daily data
        #             `divide`, `same`
        assert 'date' == df_load.columns[0],\
                'the first columns of loading data, should be `date`'

        df_load = df_load.astype(str)
        df_load['date'] = pd.to_datetime(df_load['date'])
        df_load.iloc[:, 1:] = df_load.iloc[:, 1:].apply(lambda x: x.str.replace(',', ''))
        df_load.iloc[:, 1:] = df_load.iloc[:, 1:].astype(float)

        if inputations is not None:
            days_list = df_load['date'].map(lambda x: monthrange(x.year, x.month)[1])
            for col, inputation in zip(df_load.columns[1:], inputations):
                if 'divide' == inputation:
                    df_load[col] /= days_list
            df_new = pd.DataFrame(np.repeat(df_load.to_numpy(), days_list,\
                        axis=0), columns=df_load.columns)
            df_new['date'] = pd.date_range(df_load['date'][0], periods=np.sum(days_list))
        else:
            df_new = df_load

        self.df = pd.merge(self.df, df_new, how='inner', on=['date'])

    def get_data(self, start, end):

        assert len(self.df) > 0, 'there is no data in handler, please add first!!'
        start = datetime.strptime(start, '%Y-%m-%d')
        end = datetime.strptime(end, '%Y-%m-%d')
        df_start = self.df['date'].iloc[0]
        df_end = self.df['date'].iloc[-1]
        assert (start >= df_start) & (end <= df_end), 'timestamp is out of data'

        res = self.df.loc[(self.df['date'] >= start) & (self.df['date'] <= end)]
        return res.reset_index(drop=True)

    def get_columns(self):
        return self.df.columns

    def get_start_end_tick(self):
        date = self.df['date']
        start = date.iloc[0].__format__('%Y-%m-%d')
        end = date.iloc[-1].__format__('%Y-%m-%d')
        return start, end

    def preprocess(self, columns, unit_length=1, num_input_unit=4, num_output_unit=1):

        def build_unit(dates, content, i_unit):
            dates = dates[i_unit:].reset_index(drop=True)
            content = content[:len(dates)].reset_index(drop=True)
            # content['amount(kg)'] = content['amount(kg)'].str.replace(',', '')
            content = content.astype(float)

            return pd.concat([dates, content], axis=1, sort=False, ignore_index=False)

        def serialize(dataset, num_input_unit, num_output_unit):

            feature_length = dataset.shape[0] - num_input_unit - num_output_unit + 1
            feature_x = np.zeros([feature_length, num_input_unit, dataset.shape[1]])
            for i in range(feature_length):
                for j in range(num_input_unit):
                    for k in range(dataset.shape[1]):
                        feature_x[i, j, k] = dataset.iloc[i + j, k]

            feature_y = np.zeros([feature_length, num_output_unit, 1])
            for i in range(feature_length):
                for j in range(num_output_unit):
                    feature_y[i, j, 0] = dataset.iloc[i + j + num_input_unit, -1]

            return feature_x, feature_y

        df = self.df[columns]
        df = df.dropna()

        dates = df['date']
        content = df[columns[1:]]

        units = []
        for i in range(unit_length):
            unit = build_unit(dates, content, i).set_index('date')
            units.append(unit)

        units = pd.concat(units).groupby('date').mean()
        units = units[unit_length - 1:]

        train = units[~(units.index.year == 2018) + (units.index.year == 2019)]
        valid = units[units.index.year == 2018]
        test = units[units.index.year == 2019]

        train_x, train_y = serialize(train, num_input_unit, num_output_unit)
        valid_x, valid_y = serialize(valid, num_input_unit, num_output_unit)
        test_x, test_y = serialize(test, num_input_unit, num_output_unit)

        print('tr_x: ', train_x.shape)
        print('tr_y: ', train_y.shape)
        print('va_x: ', valid_x.shape)
        print('va_y: ', valid_y.shape)
        print('te_x: ', test_x.shape)
        print('te_y: ', test_y.shape)

        np.save('features/train_x.npy', train_x)
        np.save('features/train_y.npy', train_y)
        np.save('features/valid_x.npy', valid_x)
        np.save('features/valid_y.npy', valid_y)
        np.save('features/test_x.npy', test_x)
        np.save('features/test_y.npy', test_y)

if '__main__' == __name__:

    path = './data/use'

    # value is None for daily data
    data_common = {
        'currency': None,
        'powder_feed': ['same', 'same'],
        'yellow_bean': ['same'],
    }
    data_wu = {
        'wu_export': ['divide', 'same'],
        'wu_price_perDate': None,
        # 'wu_price_perMonth': ['same', 'divide'],
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
        d.add(pd.read_csv(f'{path}/{filename}.csv'), inpu_method)
    for filename, inpu_method in data_wu.items():
        d.add(pd.read_csv(f'{path}/wu/{filename}.csv'), inpu_method)

    res = d.get_data(*d.get_start_end_tick())
    d.preprocess(d.get_columns(), 1, 7, 1)
    # print(res.shape)
    # print(res)
    # print(d.df)
    # print(d.start, d.end)

