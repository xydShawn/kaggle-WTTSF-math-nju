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

stat_prediction(num_ttw, stat, train_data)
