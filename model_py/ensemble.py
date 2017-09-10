# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


def plus_delta_relu(v, delta):
    res = np.array([max(vv + delta, 0) for vv in v])
    return res


def projection_on_simplex(v):
    sort_v = sorted(v)
    temp = np.array([sum(plus_delta_relu(sort_v, -vv)) for vv in sort_v])
    ind = (np.where(temp <= 1))[0][0]
    delta = (1 - sum(sort_v[ind:])) / (len(v) - ind)
    res = plus_delta_relu(v, delta)
    return res


def cost_and_gradient(real, pred, coef):
    comb = np.zeros(real.shape)
    for key, value in pred.items():
        comb += (value * coef[key, :])
    cost = 0
    gradient = np.zeros(coef.shape)
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):
            if abs(comb[i, j] + real[i, j]) >= 0.01:
                cost += (np.abs(comb[i, j] - real[i, j]) / (comb[i, j] + real[i, j]))
                for k in range(coef.shape[0]):
                    if (comb[i, j] - real[i, j]) >= 0:
                        gradient[k, j] += ((2 * pred[k][i, j] * real[i, j]) / ((comb[i, j] + real[i, j]) ** 2))
                    else:
                        gradient[k, j] += ((2 * pred[k][i, j] * comb[i, j]) / ((comb[i, j] + real[i, j]) ** 2))
    temp = 2 / (real.shape[0] * real.shape[1])
    cost *= temp
    gradient *= temp
    return cost, gradient


def stat_prediction_all(data, num_ttw, stat_func):
    sub_columns = data.columns[-num_ttw:]
    sub_data = data[sub_columns]
    sub_data = sub_data.fillna(0)
    print('preprocessing done!')

    sub_data = sub_data.values
    result = np.zeros((len(data), 60))
    xx = np.zeros(len(data))
    for i in tqdm(range(len(data))):
        ind = np.where(sub_data[i] != 0)
        if len(ind[0]) == 0:
            xx[i] = 0
        else:
            xx[i] = stat_func(sub_data[i, ind[0][0]:])

    for i in range(60):
        result[:, i] = xx
    return result


def stat_prediction_weekend(data, num_ttw, stat_func):
    train_date = pd.read_csv('../tmp/train_date_class.csv').iloc[(-num_ttw - 60):(-60)]
    test_date = pd.read_csv('../tmp/train_date_class.csv').iloc[-60:]
    sub_columns_workday = (data.columns[-num_ttw:])[train_date.is_weekend == 0]
    sub_columns_weekend = (data.columns[-num_ttw:])[train_date.is_weekend == 1]
    print('length of workday: %d' % (len(sub_columns_workday)))
    print('length of weekend: %d' % (len(sub_columns_weekend)))
    sub_data_workday = data[sub_columns_workday]
    sub_data_weekend = data[sub_columns_weekend]
    print('shape of workday data: (%d, %d)' % (sub_data_workday.values.shape[0], sub_data_workday.values.shape[1]))
    print('shape of weekend data: (%d, %d)' % (sub_data_weekend.values.shape[0], sub_data_weekend.values.shape[1]))
    sub_data_workday = sub_data_workday.fillna(0)
    sub_data_weekend = sub_data_weekend.fillna(0)
    print('preprocessing done!')

    sub_data_workday = sub_data_workday.values
    sub_data_weekend = sub_data_weekend.values
    result = np.zeros((len(data), 60))
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

    for i in range(60):
        if test_date.iloc[i]['is_weekend'] == 1:
            result[:, i] = xx_weekend
        else:
            result[:, i] = xx_workday
    return result


def stat_prediction_day_by_day(data, num_ttw, stat_func):
    train_date = pd.read_csv('../tmp/train_date_class.csv').iloc[(-num_ttw - 60):(-60)]
    test_date = pd.read_csv('../tmp/train_date_class.csv').iloc[-60:]
    sub_columns = dict()
    sub_data = dict()
    for i in range(7):
        sub_columns[i] = (data.columns[-num_ttw:])[train_date.weekday == i]
        print('days of weekday %d: %d'.format(i + 1, len(sub_columns[i])))
        sub_data[i] = data[sub_columns[i]]
        print('shape of data of weekday %d: %s'.format(i + 1, str(sub_data[i].values.shape)))
        sub_data[i] = sub_data[i].fillna(0)
        sub_data[i] = sub_data[i].values
    print('preprocessing done!')

    result = np.zeros((len(data), 60))
    xx = dict()
    for i in range(7):
        xx[i] = np.zeros(len(data))
        for j in tqdm(range(len(data))):
            ind = np.where(sub_data[i][j] != 0)
            if len(ind[0]) == 0:
                xx[i][j] = 0
            else:
                xx[i][j] = stat_func(sub_data[i][j, ind[0][0]:])

    for i in range(60):
        result[:, i] = xx[test_date.iloc[i, 'weekday']]
    return result


ts_result = pd.read_csv('../result/ts_for_all_490.csv', header=None).values
prophet_result = pd.read_csv('../result/prophet_for_all_490.csv', header=None).values
total_data = pd.read_csv('../data/clean_NaN_nearest_int.csv.gz', header=None, compression='gzip')

shuffle_ind = np.arange(len(total_data))
np.random.shuffle(shuffle_ind)
total_data_shuffle = total_data.iloc[list(shuffle_ind)]
ts_result = ts_result[list(shuffle_ind), :]
prophet_result = prophet_result[list(shuffle_ind), :]

train_size = 20000
train_data = total_data_shuffle.iloc[:train_size, :-60]
real_result = total_data_shuffle.iloc[:train_size, -60:]
real_result = real_result.fillna(0)
real_result = real_result.values

pred_result = dict()
pred_result[0] = ts_result[:train_size, :]
pred_result[1] = prophet_result[:train_size, :]
pred_result[2] = stat_prediction_all(train_data, 56, np.min)
pred_result[3] = stat_prediction_all(train_data, 56, np.max)
pred_result[4] = stat_prediction_all(train_data, 56, np.median)
pred_result[5] = stat_prediction_all(train_data, 56, np.mean)
pred_result[6] = stat_prediction_weekend(train_data, 49, np.min)
pred_result[7] = stat_prediction_weekend(train_data, 49, np.max)
pred_result[8] = stat_prediction_weekend(train_data, 49, np.median)
pred_result[9] = stat_prediction_weekend(train_data, 49, np.mean)


print('complete fetching the result')

# parameter
epochs = 10
batch_size = 200
steps = train_size // batch_size
alpha = 0.01

# initialization
coef = np.ones((10, 60)) / 10

for i in range(epochs):
    for j in tqdm(range(steps)):
        batch_pred = dict()
        for key, value in pred_result.items():
            batch_pred[key] = value[(200 * j):(200 * (j + 1)), :]
        batch_real = real_result[(200 * j):(200 * (j + 1)), :]
        batch_cost, batch_gradient = cost_and_gradient(batch_real, batch_pred, coef)
        coef -= (alpha * batch_gradient)
        for k in range(60):
            coef[:, k] = projection_on_simplex(coef[:, k])
        if j == (steps - 1):
            print('epoch: %d, cost: %f' % (i + 1, batch_cost))

print('train completed')

print(coef.sum(axis=0))

val_data = total_data_shuffle.iloc[train_size:, :-60]
val_real_result = total_data_shuffle.iloc[train_size:, -60:]
val_real_result = val_real_result.fillna(0)
val_real_result = val_real_result.values
 
val_pred_result = dict()
val_pred_result[0] = ts_result[train_size:, :]
val_pred_result[1] = prophet_result[train_size:, :]
val_pred_result[2] = stat_prediction_all(val_data, 56, np.min)
val_pred_result[3] = stat_prediction_all(val_data, 56, np.max)
val_pred_result[4] = stat_prediction_all(val_data, 56, np.median)
val_pred_result[5] = stat_prediction_all(val_data, 56, np.mean)
val_pred_result[6] = stat_prediction_weekend(val_data, 49, np.min)
val_pred_result[7] = stat_prediction_weekend(val_data, 49, np.max)
val_pred_result[8] = stat_prediction_weekend(val_data, 49, np.median)
val_pred_result[9] = stat_prediction_weekend(val_data, 49, np.mean)
val_cost, _ = cost_and_gradient(val_real_result, val_pred_result, coef)
print('cost on validation set: %f' % (val_cost))

