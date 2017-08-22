# -*- coding: utf-8 -*-

import time

import numpy as np
import pandas as pd
import tensorflow as tf


data = pd.read_csv('../team/kaggle-WTTSF-math-nju/data/clean_NaN_nearest_int.csv.gz', header=None, compression='gzip')
train_data = data.values[:, -180:-60]
test_data = data.values[:, -120:]
train_data = train_data / (train_data.max(axis=1).reshape(-1, 1) + 0.001)
test_data = test_data / (test_data.max(axis=1).reshape(-1, 1) + 0.001)
print(train_data.shape)
print(test_data.shape)

learning_rate = 0.01
batch_size = 128

n_input = 60
n_steps = 60
n_hidden = 256

x = tf.placeholder(tf.float32, [None, n_steps + n_input - 1])
y = tf.placeholder(tf.float32, [None, n_steps])

def RNN(x, n_steps, n_input, n_hidden, batch_size):
    # input_gate
    ix = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, n_hidden]))
    # forget_gate
    fx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, n_hidden]))
    # memory
    cx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, n_hidden]))
    # output_gate
    ox = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, n_hidden]))
    # weight and biases
    w = tf.Variable(tf.truncated_normal([n_hidden, 1]))
    b = tf.Variable(tf.zeros([1]))

    def lstm_cell(ii, hh, cc):
        input_gate = tf.sigmoid(tf.matmul(ii, ix) + tf.matmul(hh, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(ii, fx) + tf.matmul(hh, fm) + fb)
        update = tf.tanh(tf.matmul(ii, cx) + tf.matmul(hh, cm) + cb)
        cc = forget_gate * cc + input_gate * update
        output_gate = tf.sigmoid(tf.matmul(ii, ox) + tf.matmul(hh, om) + ob)
        return output_gate * tf.tanh(cc), cc

    cc = tf.Variable(tf.zeros([batch_size, n_hidden]))
    hh = tf.Variable(tf.zeros([batch_size, n_hidden]))
    hh, cc = lstm_cell(tf.slice(x, begin=[0, 0], size=[batch_size, n_input]), hh, cc)
    pred = tf.matmul(hh, w) + b
    for i in range(1, n_steps):
        hh, cc = lstm_cell(tf.slice(x, begin=[0, i], size=[batch_size, n_input]), hh, cc)
        HH = tf.matmul(hh, w) + b
        pred = tf.concat([pred, HH], 1)
    pred = tf.clip_by_value(pred, 0.0, 1.0)
    return pred

pred = RNN(x, n_steps, n_input, n_hidden, batch_size)
cost = tf.reduce_mean(2.0 * tf.abs(pred - y) / (pred + y + 0.00000001))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
start_index = 0
for step in range(100000):
    if start_index + batch_size > train_data.shape[0]:
        if start_index == train_data.shape[0]:
            batch_x = train_data[0:batch_size, :-1]
            batch_y = train_data[0:batch_size, -60:]
            start_index = batch_size
        else:
            temp = start_index + batch_size - train_data.shape[0]
            batch_x = np.vstack((train_data[start_index:, :-1], train_data[:temp, :-1]))
            batch_y = np.vstack((train_data[start_index:, -60:], train_data[:temp, -60:]))
            start_index = temp
    else:
        temp = start_index + batch_size
        batch_x = train_data[start_index:temp, :-1]
        batch_y = train_data[start_index:temp, -60:]
        start_index = temp
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
    if (step + 1) % 100 == 0:
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print('Iter' + str(step + 1) + ', batch loss: ' + '{:.4f}'.format(loss))

print('Optimization Finished!')

n_batches = int(train_data.shape[0] / batch_size)
losses = list()
for i in range(n_batches):
    batch_x = test_data[(i * batch_size):((i + 1) * batch_size), :-1]
    batch_y = test_data[(i * batch_size):((i + 1) * batch_size), -60:]
    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
    losses.append(loss)
print('Total loss: {0}'.format(np.mean(np.array(losses))))

sess.close()
