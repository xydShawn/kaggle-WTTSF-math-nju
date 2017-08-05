# -*- coding: utf-8 -*-

import argparse
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

def stat_prediction_weekend(num_ttw, stat, train_data):
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
        xx_workday = sub_data_workday.mean(axis=1)
        xx_weekend = sub_data_weekend.mean(axis=1)
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


def stat_prediction_day_by_day(num_ttw, stat, train_data):
    train_date = pd.read_csv('../tmp/train_date_class.csv').iloc[-num_ttw:]
    test_date = pd.read_csv('../tmp/test_date_class.csv')
    sub_columns = {}
    sub_data = {}
    for i in range(7):
        sub_columns[i] = (train_data.columns[-num_ttw:])[train_date.weekday == i]
        print('days of weekday %d: %d' % (i + 1, len(sub_columns[i])))
        sub_data[i] = train_data[sub_columns[i]]
        print('shape of data of weekday %d: %s' % (i + 1, str(sub_data[i].values.shape)))
        sub_data[i] = sub_data[i].fillna(0)
    print('preprocessing done')

    result = np.zeros((len(train_data), 60))
    xx = {}
    if stat == 'median':
        for i in range(7):
            xx[i] = sub_data[i].median(axis=1)
    elif stat == 'mean':
        for i in range(7):
            xx[i] = sub_data[i].mean(axis=1)
    else:
        print('to do')
        return

    for i in range(60):
        result[:, i] = xx[test_date.loc[i, 'weekday']]
    print('completing')
    print('shape of result: %s' % (str(result.shape)))

    generate_submission_1(result)   


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num', help='length of time window, default=56', default=56, type=int)
parser.add_argument('-s', '--stat', help='choose statistics, default: median', default='median', type=str)
parser.add_argument('-d', '--datatype', help='data type, original or treated, default: treated', default='treated', type=str)
parser.add_argument('-m', '--method', help='choose method', default=0, type=int)
args = parser.parse_args()

num_ttw = args.num
stat = args.stat
data_type = args.datatype
method = args.method
print('length of time window: %d' % (num_ttw))
print('statistics: %s' % (stat))
print('data type: %s' % (data_type))
print('choose method: %d' % (method))

if data_type == 'oringinal':
    train_data = pd.read_csv('../data/train_1.csv')
    print('fetch original data')
elif data_type == 'treated':
    train_data = pd.read_csv('../data/clean_NaN_nearest_int.csv', header=None)
    print('fetch treated data')
else:
    print('to do')

if method == 0:
    stat_prediction(num_ttw, stat, train_data)
elif method == 1:
    stat_prediction_weekend(num_ttw, stat, train_data)
elif method == 2:
    stat_prediction_day_by_day(num_ttw, stat, train_data)
else:
    print('to do')
