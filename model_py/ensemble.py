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


def cost(real, pred, coef):
    comb = np.zeros(real.shape)
    for key, value in pred.items():
        comb += (value * coef[key, :])
    cost = 0
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):
            if abs(comb[i, j] + real[i, j]) >= 0.5:
                cost += (np.abs(comb[i, j] - real[i, j]) / (comb[i, j] + real[i, j]))
    temp = 2 / (real.shape[0] * real.shape[1])
    cost *= temp
    return cost


def cost_and_gradient(real, pred, coef):
    comb = np.zeros(real.shape)
    for key, value in pred.items():
        comb += (value * coef[key, :])
    cost = 0
    gradient = np.zeros(coef.shape)
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):
            if abs(comb[i, j] + real[i, j]) >= 0.5:
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

shuffle_ind = np.arange(len(total_data))
np.random.shuffle(shuffle_ind)
total_data_shuffle = total_data[list(shuffle_ind), :]
# ts_result = ts_result[list(shuffle_ind), :]
# prophet_result = prophet_result[list(shuffle_ind), :]

train_size = 60000
train_data = total_data_shuffle[:train_size, :-62]
real_result = total_data_shuffle[:train_size, -62:]

temp_date = pd.read_csv('../tmp/train_date_class_2.csv')
train_weekend = temp_date.iloc[:-62]['is_weekend'].values
test_weekend = temp_date.iloc[-62:]['is_weekend'].values
train_weekday = temp_date.iloc[:-62]['weekday'].values
test_weekday = temp_date.iloc[-62:]['weekday'].values

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

# parameter
epochs = 100
batch_size = 200
steps = train_size // batch_size
alpha = 0.2

# initialization
coef = np.zeros((len(pred_result), 62))
coef[6, :] = 1.

for i in range(epochs):
    """
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
    """
    train_cost, train_gradient = cost_and_gradient(real_result, pred_result, coef)
    coef -= (alpha * train_gradient)
    for k in range(62):
        coef[:, k] = projection_on_simplex(coef[:, k])
    print('epoch: %d, cost: %f' % (i + 1, train_cost))
    if (i + 1) % 30 == 0:
        alpha *= 0.5
        print('learning rate: %f' % (alpha))


print('train completed')

for i in range(62):
    print(coef[:, i])
print(coef.sum(axis=0))
coef_pd = pd.DataFrame(coef)
coef_pd.to_csv('../result/coef.csv', index=False)

val_data = total_data_shuffle[train_size:, :-62]
val_real_result = total_data_shuffle[train_size:, -62:]
 
val_pred_result = dict()
# val_pred_result[0] = ts_result[train_size:, :]
# val_pred_result[1] = prophet_result[train_size:, :]
val_pred_result[0] = stat_prediction_all(val_data, 56, 62, np.min)
val_pred_result[1] = stat_prediction_all(val_data, 56, 62, np.max)
val_pred_result[2] = stat_prediction_all(val_data, 56, 62, np.median)
val_pred_result[3] = stat_prediction_all(val_data, 56, 62, np.mean)
val_pred_result[4] = stat_prediction_weekend(val_data, 49, 62, np.min, train_weekend, test_weekend)
val_pred_result[5] = stat_prediction_weekend(val_data, 49, 62, np.max, train_weekend, test_weekend)
val_pred_result[6] = stat_prediction_weekend(val_data, 49, 62, np.median, train_weekend, test_weekend)
val_pred_result[7] = stat_prediction_weekend(val_data, 49, 62, np.mean, train_weekend, test_weekend)
val_cost = cost(val_real_result, val_pred_result, coef)
print('cost on validation set: %f' % (val_cost))

