# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from utils import *



def stat_prediction(num_ttw, stat, train_data):
    sub_columns = train_data.columns[-num_ttw:]
    sub_data = train_data[sub_columns]
    sub_data_pre = sub_data.fillna(0)
    print('preprocessing done')

    result = np.zeros((len(train_data), 60))

    if stat == 'median':
        xx = sub_data_pre.median(axis=1)
    elif stat == 'mean':
        xx = sub_data_pre.mean(axis=1)
    else:
        print('to do')
        return

    for i in range(60):
        result[:, i] = xx
    print('completing')
    print('shape of result', result.shape)

    generate_submission_1(result)

def stat_prediction_weekday(num_ttw, stat, train_data):
    train_date = pd.read_csv('../tmp/train_date_class.csv').iloc[-num_ttw:]
    test_date = pd.read_csv('../tmp/test_date_class.csv')
    sub_columns_workday = (train_data.columns[-num_ttw:])[train_date.is_weekend == 0]
    sub_columns_weekend = (train_data.columns[-num_ttw:])[train_date.is_weekend == 1]
    print('length of workday: ', len(sub_columns_workday))
    print('length of weekend: ', len(sub_columns_weekend))
    sub_data_workday = train_data[sub_columns_workday]
    sub_data_weekend = train_data[sub_columns_weekend]
    print('shape of workday data: ', sub_data_workday.values.shape)
    print('shape of weekend data: ', sub_data_weekend.values.shape)
    sub_data_workday = sub_data_workday.fillna(0)
    sub_data_weekend = sub_data_weekend.fillna(0)
    print('preprocessing done')

    result = np.zeros((len(train_data), 60))
    if stat == 'median':
        xx_workday = sub_data_workday.median(axis=1)
        xx_weekend = sub_data_weekend.median(axis=1)
    elif stat == 'mean':
        xx_workday = sub_data_workday.median(axis=1)
        xx_weekend = sub_data_weekend.median(axis=1)
    else:
        print('to do')
        return

    for i in range(60):
        if test_date.loc[i, 'is_weekend'] == 1:
            result[:, i] = xx_weekend
        else:
            result[:, i] = xx_workday
    print('completing')
    print('shape of result', result.shape)
    print('type of result', result.dtype)

    generate_submission_1(result)

num_ttw = 56
stat = 'median'
data_type = 'treated'
if data_type == 'oringinal':
    train_data = pd.read_csv('../data/train_1.csv')
    print('fetch original data')
elif data_type == 'treated':
    train_data = pd.read_csv('../data/clean_NaN_nearest_int.csv', header=None)
    print('fetch treated data')
else:
    print('to do')

stat_prediction_weekday(num_ttw, stat, train_data)
