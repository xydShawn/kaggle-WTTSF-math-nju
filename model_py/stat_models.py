# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


def cost(real, pred):
    loss = 0
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):
            if abs(pred[i, j] + real[i, j]) >= 0.5:
                loss += (np.abs(pred[i, j] - real[i, j]) / (pred[i, j] + real[i, j]))
    temp = 2 / (real.shape[0] * real.shape[1])
    loss *= temp
    return loss


def stat_prediction_all(data, train_ttw, pred_ttw, stat_func):
    sub_data = data[:, -train_ttw:]
    result = np.zeros((len(data), pred_ttw))
    xx = np.zeros(len(data))
    for i in tqdm(range(len(data))):
        ind = np.where(sub_data[i] != 0)
        if len(ind[0]) == 0:
            xx[i] = 0
        else:
            xx[i] = stat_func(sub_data[i, ind[0][0]:])

    for i in range(pred_ttw):
        result[:, i] = xx
    return result


def stat_prediction_weekend(data, train_ttw, pred_ttw, stat_func, train_weekend, test_weekend):
    train_weekend = train_weekend[-train_ttw:]
    sub_data_workday = data[:, -train_ttw:][:, train_weekend==0]
    sub_data_weekend = data[:, -train_ttw:][:, train_weekend==1]
    print('shape of workday data: (%d, %d)' % (sub_data_workday.shape[0], sub_data_workday.shape[1]))
    print('shape of weekend data: (%d, %d)' % (sub_data_weekend.shape[0], sub_data_weekend.shape[1]))
    result = np.zeros((len(data), pred_ttw))
    xx_workday = np.zeros(len(data))
    xx_weekend = np.zeros(len(data))
    for i in tqdm(range(len(data))):
        ind1 = np.where(sub_data_workday[i] != 0)
        if len(ind1[0]) == 0:
            xx_workday[i] = 0
        else:
            xx_workday[i] = stat_func(sub_data_workday[i, ind1[0][0]:])
        ind2 = np.where(sub_data_weekend[i] != 0)
        if len(ind2[0]) == 0:
            xx_weekend[i] = 0
        else:
            xx_weekend[i] = stat_func(sub_data_weekend[i, ind2[0][0]:])

    for i in range(pred_ttw):
        if test_weekend[i] == 1:
            result[:, i] = xx_weekend
        else:
            result[:, i] = xx_workday
    return result


def stat_prediction_day_by_day(data, train_ttw, pred_ttw, stat_func, train_weekday, test_weekday):
    train_weekday = train_weekday[-num_ttw:]
    sub_data = dict()
    for i in range(7):
        sub_data[i] = data[:, -num_ttw:][:, train_weekday==i]
        print('shape of data of weekday %d: %s' % (i + 1, str(sub_data[i].shape)))

    result = np.zeros((len(data), pred_ttw))
    xx = dict()
    for i in range(7):
        xx[i] = np.zeros(len(data))
        for j in tqdm(range(len(data))):
            ind = np.where(sub_data[i][j] != 0)
            if len(ind[0]) == 0:
                xx[i][j] = 0
            else:
                xx[i][j] = stat_func(sub_data[i][j, ind[0][0]:])

    for i in range(pred_ttw):
        result[:, i] = xx[test_weekday[i]]
    return result


# ts_result = pd.read_csv('../result/ts_for_all_490.csv', header=None).values
# prophet_result = pd.read_csv('../result/prophet_for_all_490.csv', header=None).values
total_data = pd.read_csv('../data/clean_NaN_linear_2.csv', header=None)
total_data = total_data.fillna(0)
total_data = total_data.values
train_data = total_data[:, :-62]
real_result = total_data[:, -62:]

temp_date = pd.read_csv('../tmp/train_date_class_2.csv')
train_weekend = temp_date.iloc[:-62]['is_weekend'].values
test_weekend = temp_date.iloc[-62:]['is_weekend'].values
train_weekday = temp_date.iloc[:-62]['weekday'].values
test_weekday = temp_date.iloc[-62:]['weekday'].values

'''
pred_result = dict()
# pred_result[0] = ts_result[:train_size, :]
# pred_result[1] = prophet_result[:train_size, :]
pred_result[0] = stat_prediction_all(train_data, 56, 62, np.min)
pred_result[1] = stat_prediction_all(train_data, 56, 62, np.max)
pred_result[2] = stat_prediction_all(train_data, 56, 62, np.median)
pred_result[3] = stat_prediction_all(train_data, 56, 62, np.mean)
pred_result[4] = stat_prediction_weekend(train_data, 49, 62, np.min, train_weekend, test_weekend)
pred_result[5] = stat_prediction_weekend(train_data, 49, 62, np.max, train_weekend, test_weekend)
pred_result[6] = stat_prediction_weekend(train_data, 49, 62, np.median, train_weekend, test_weekend)
pred_result[7] = stat_prediction_weekend(train_data, 49, 62, np.mean, train_weekend, test_weekend)

print('complete fetching the result')

for key, value in pred_result.items():
    loss = cost(real_result, value)
    print('result%d: %f' % (key, loss))

comb = (pred_result[2] + pred_result[6]) / 2
print(cost(real_result, comb))
'''

pred_result = np.zeros((7, len(train_data), 62))
ttw = [11, 18, 30, 48, 126, 203, 329]
print(ttw)
for i, t in enumerate(ttw):
    pred_result[i] = stat_prediction_weekend(train_data, t, 62, np.median, train_weekend, test_weekend)

result_1 = np.median(pred_result, axis=0)
print(result_1.shape)
print('cost of result_1: %f' % (cost(real_result, result_1)))

result_2 = stat_prediction_weekend(train_data, 49, 62, np.median, train_weekend, test_weekend)
print(result_2.shape)
print('cost of result_2: %f' % (cost(real_result, result_2)))

result_3 = stat_prediction_all(train_data, 56, 62, np.median)
print(result_3.shape)
print('cost of result_3: %f' % (cost(real_result, result_3)))

comb = result_1 * 0.5 + result_2 * 0.3 + result_2 * 0.2
comb = np.round(comb)
print('cost of combined result: %f' % (cost(real_result, comb)))
