# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-12 16:25:45
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-10-12 18:01:23

import numpy as np
import pandas as pd
import theano as th
import theano.tensor as T
import gc

from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams


def load_data(filename, nrows=None):
    """load data from file, first column as index, dtype=float32"""
    ind = pd.read_csv(filename, sep=' ', header=None, index_col=0, nrows=5)
    dtype_dict = {c: np.float32 for c in ind.columns}
    data = pd.read_csv(filename, sep=' ', header=None, index_col=0,
                       dtype=dtype_dict, nrows=nrows)
    # normalize
    data = (data - data.mean()) / data.std()
    gc.collect()

    return data


def load_label(filename):
    label_data = pd.read_csv(filename, header=None, index_col=0)
    label_map = {}
    for ind, lab in enumerate(np.unique(label_data.values)):
        label_map[lab] = ind

    label_data = label_data.applymap(lambda x: label_map[x])
    gc.collect()

    return label_data, label_map


def random_number(shape, scale=1):
    return (scale * np.random.randn(*shape)).astype('float32')


def zero_number(shape):
    return np.zeros(shape).astype('float32')


def initialize_NNet(n_input, n_output, archi=128,
                    n_hid_layers=3, scale=0.033):
    """initialize the NNet paramters, archi: hidden layer neurons"""
    Ws = []
    bs = []
    cache_Ws = []
    cache_bs = []

    # input layer
    Ws.append(th.shared(random_number([n_input, archi], scale=scale)))
    cache_Ws.append(th.shared(zero_number((n_input, archi))))

    bs.append(th.shared(random_number([archi], scale=scale)))
    cache_bs.append(th.shared(zero_number(archi)))

    # hidden layers
    for i in range(n_hid_layers):
        Ws.append(th.shared(random_number([archi / 2, archi], scale=scale)))
        cache_Ws.append(th.shared(zero_number((archi / 2, archi))))

        bs.append(th.shared(random_number([archi], scale=scale)))
        cache_bs.append(th.shared(zero_number(archi)))

    # output layer
    Ws.append(th.shared(random_number([archi / 2, n_output], scale=scale)))
    cache_Ws.append(th.shared(zero_number((archi / 2, n_output))))

    bs.append(th.shared(random_number([n_output], scale=scale)))
    cache_bs.append(th.shared(zero_number(n_output)))

    return Ws, bs, cache_Ws, cache_bs


def maxout(Z, stop_dropout, archi, dropout_rate, seed=5432):
    th.config.floatX = 'float32'
    Z_out = T.maximum(Z[:, :int(archi / 2)], Z[:, int(archi / 2):])
    prob = (1 - dropout_rate)
    srng = RandomStreams(seed=seed)

    return ifelse(T.lt(stop_dropout, 1.05),
                  Z_out * srng.binomial(size=T.shape(Z_out),
                                        p=prob).astype('float32'),
                  Z_out)


def softmax(z):
    Z = T.exp(z)
    results, _ = th.scan(lambda x: x / T.sum(x), sequences=Z)
    return results


def update(para, grad, moment_cache, lr, moment):
    """theano update auxiliary function: use SGD plus momentum"""
    param_update = []
    cache_update = []

    for ix in range(len(grad)):
        change = moment * moment_cache[ix] - lr * grad[ix]
        param_update.append((para[ix], para[ix] + change))
        cache_update.append((moment_cache[ix], change))

    return param_update + cache_update


def gen_y_hat(i, n_output, data, label_data, cache):
    """give the np array of y_hat"""
    try:
        return cache[i]

    except KeyError:
        y_h = np.zeros(n_output, dtype=np.float32)
        y_h[label_data[1].loc[data.index[i]]] = 1
        cache[i] = y_h

        return cache[i]


def accuracy(from_ind, to_ind, data, forward, n_output, label_data, cache):
    """compute the accuracy of the model"""
    X = []
    y = []

    for ind in range(from_ind + 4, to_ind - 4):
        X.append(data.iloc[(ind - 4):(ind + 5)].values.ravel())
        y.append(gen_y_hat(ind, n_output, data, label_data, cache))

    # stop_dropout > 1.05 the model won't do dropout
    y_pred = forward(X, 1.25)
    match = 0
    for i, ind in enumerate(range(from_ind + 4, to_ind - 4)):
        if np.argmax(y_pred[i]) == label_data[1].iloc[ind]:
            match += 1

    return match / len(y_pred)
