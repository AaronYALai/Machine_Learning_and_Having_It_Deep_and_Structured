# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-12 16:25:45
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-03 17:24:15

import numpy as np
import pandas as pd
import theano as th
import theano.tensor as T
import gc

from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams


def load_data(filename, nrows=None, normalize=True):
    """load data from file, first column as index, dtype=float32"""
    ind = pd.read_csv(filename, sep=' ', header=None, index_col=0, nrows=5)
    dtype_dict = {c: np.float32 for c in ind.columns}
    data = pd.read_csv(filename, sep=' ', header=None, index_col=0,
                       dtype=dtype_dict, nrows=nrows)
    # normalize
    if normalize:
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


def make_data(data, prob_file, label_data=None):
    prob_data = np.load(prob_file)
    df = pd.DataFrame(data=prob_data, index=data.index)
    speakers = list(set(['_'.join(name.split('_')[:2]) for name in df.index]))

    X = {}
    labels = {}
    for speaker in speakers:
        speaker_indexes = df.index.str.startswith(speaker)
        X[speaker] = df.iloc[speaker_indexes].values
        if label_data is not None:
            labels[speaker] = label_data.iloc[speaker_indexes].values

    return X, labels


def random_number(shape, scale=1):
    return (scale * np.random.randn(*shape)).astype('float32')


def zero_number(shape):
    return np.zeros(shape).astype('float32')


def initialize_RNN(n_input, n_output, archi=128,
                   n_hid_layers=3, scale=0.033):
    Ws = []
    bs = []
    cache_Ws = []
    cache_bs = []

    Ws.append(th.shared(random_number([n_input, archi], scale=scale)))
    cache_Ws.append(th.shared(zero_number((n_input, archi))))

    bs.append(th.shared(random_number([archi], scale=scale)))
    cache_bs.append(th.shared(zero_number(archi)))

x_seq = T.fmatrix()
y_hat = T.fmatrix()
ind = T.scalar()    #Help to do minibatch
bud = T.scalar()    #Help to do Dropout 

cons = 0.001; a=0.0; s=0.01; neuron = 160

a_0 = th.shared(0*np.random.randn(neuron))
Wi = th.shared(s*np.random.randn(48,neuron))
bi = th.shared(cons*np.random.randn(neuron)-a)

Wh = th.shared(s*np.identity(neuron)-a)
Wof = th.shared(s*np.random.randn(2*neuron,neuron)-a)
Wob = th.shared(s*np.random.randn(2*neuron,neuron)-a)
bh = th.shared(cons*np.random.randn(neuron)-a)
bof = th.shared(cons*np.random.randn(neuron)-a)
bob = th.shared(cons*np.random.randn(neuron)-a)
"""
W2h = th.shared(s*np.identity(neuron)-a)
W2of = th.shared(s*np.random.randn(2*neuron,neuron)-a)
W2ob = th.shared(s*np.random.randn(2*neuron,neuron)-a)
b2h = th.shared(cons*np.random.randn(neuron)-a)
b2of = th.shared(cons*np.random.randn(neuron)-a)
b2ob = th.shared(cons*np.random.randn(neuron)-a)
"""
W3h = th.shared(s*np.identity(neuron)-a)
W3o = th.shared(s*np.random.randn(2*neuron,48)-a)
b3h = th.shared(cons*np.random.randn(neuron)-a)
b3o = th.shared(cons*np.random.randn(48)-a)

Auxiliary = []; Temp = []
parameters = [Wi,bi,Wh,Wof,Wob,bh,bof,bob,W3h,W3o,b3h,b3o]#W2h,W2of,W2ob,b2h,b2of,b2ob,
for param in parameters:
    Auxiliary.append(th.shared(np.zeros(param.get_value().shape)))
    Temp.append(th.shared(np.zeros(param.get_value().shape)))
    
c = 5
Wi = th.gradient.grad_clip(Wi,-c,c)
bi = th.gradient.grad_clip(bi,-c,c)
Wh = th.gradient.grad_clip(Wh,-c,c)
Wof = th.gradient.grad_clip(Wof,-c,c)
Wob = th.gradient.grad_clip(Wob,-c,c)
bh = th.gradient.grad_clip(bh,-c,c)
bof = th.gradient.grad_clip(bof,-c,c)
bob = th.gradient.grad_clip(bob,-c,c)
W3h = th.gradient.grad_clip(W3h,-c,c)
W3o = th.gradient.grad_clip(W3o,-c,c)
b3h = th.gradient.grad_clip(b3h,-c,c)
b3o = th.gradient.grad_clip(b3o,-c,c) 

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


def accuracy(from_ind, to_ind, data, forward, n_output, label_data,
             cache, save_pred=False, save_name='pred_prob'):
    """compute the accuracy of the model"""
    X = []
    y = []

    for ind in range(from_ind, to_ind):
        if ind < from_ind + 4:
            sils = np.zeros((from_ind + 4 - ind) * data.shape[1])
            dat = data.iloc[from_ind:(ind + 5)].values.ravel()
            X.append(np.concatenate((sils, dat)))

        elif ind > (to_ind - 5):
            dat = data.iloc[(ind - 4):to_ind].values.ravel()
            sils = np.zeros((5 - to_ind + ind) * data.shape[1])
            X.append(np.concatenate((dat, sils)))

        else:
            X.append(data.iloc[(ind - 4):(ind + 5)].values.ravel())

        y.append(gen_y_hat(ind, n_output, data, label_data, cache))

    # stop_dropout > 1.05 the model won't do dropout
    y_pred = forward(X, 1.25)
    if save_pred:
        np.save(save_name, y_pred)

    match = 0
    for i, ind in enumerate(range(from_ind + 4, to_ind - 4)):
        if np.argmax(y_pred[i]) == label_data[1].iloc[ind]:
            match += 1

    return match / len(y_pred)
