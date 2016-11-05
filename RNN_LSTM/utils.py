# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-12 16:25:45
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-05 20:39:56

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


def identity_mat(N, scale):
    return (scale * np.identity(N)).astype('float32')


def initialize_RNN(n_input, n_output, archi=128,
                   n_hid_layers=2, scale=0.033, scale_b=0.001,
                   clip_thres=5):
    W_in_out = []
    W_out_forward = []
    W_out_backward = []
    W_memory = []

    b_in_out = []
    b_out_forward = []
    b_out_backward = []
    b_memory = []

    # initial memory
    a_0 = th.shared(random_number([archi], 0))

    # input layer
    W_in_out.append(th.shared(random_number([n_input, archi], scale)))
    b_in_out.append(th.shared(random_number([archi], scale_b)))

    # hidden layers
    for i in range(n_hid_layers):
        # initialize memory weights as identity matrix
        W_memory.append(th.shared(identity_mat(archi, scale)))
        W_out_forward.append(th.shared(random_number([2*archi, archi], scale)))
        W_out_backward.append(th.shared(random_number([2*archi, archi], scale)))

        b_memory.append(th.shared(random_number([archi], scale_b)))
        b_out_forward.append(th.shared(random_number([archi], scale_b)))
        b_out_backward.append(th.shared(random_number([archi], scale_b)))

    W_memory.append(th.shared(identity_mat(archi, scale)))
    b_memory.append(th.shared(random_number([archi], scale_b)))

    # output layer
    W_in_out.append(th.shared(random_number([2 * archi, n_output], scale)))
    b_in_out.append(th.shared(random_number([archi], scale_b)))

    param_Ws = [W_in_out, W_out_forward, W_out_backward, W_memory]
    param_bs = [b_in_out, b_out_forward, b_out_backward, b_memory]

    # help to do advanced optimization (ex. NAG, RMSProp)
    aux_Ws = []
    aux_bs = []

    # help to do mini-batch update (to store gradients)
    cache_Ws = []
    cache_bs = []

    for i in range(4):
        aux_W = []
        aux_b = []
        cache_W = []
        cache_b = []

        for j in range(len(param_Ws[i])):
            W_shape = param_Ws[i][j].get_value().shape
            b_shape = param_bs[i][j].get_value().shape

            aux_W.append(th.shared(zero_number(W_shape)))
            aux_b.append(th.shared(zero_number(b_shape)))

            cache_W.append(th.shared(zero_number(W_shape)))
            cache_b.append(th.shared(zero_number(b_shape)))

            # set the restricted numerical range for gradient values
            param_Ws[i][j] = th.gradient.grad_clip(param_Ws[i][j],
                                                   -clip_thres, clip_thres)

            param_bs[i][j] = th.gradient.grad_clip(param_bs[i][j],
                                                   -clip_thres, clip_thres)

        aux_Ws.append(aux_W)
        aux_bs.append(aux_b)

        cache_Ws.append(cache_W)
        cache_bs.append(cache_b)

    return param_Ws, param_bs, aux_Ws, aux_bs, cache_Ws, cache_bs, a_0


def tanh(Z):
    exp_m2z = T.exp(-2 * Z)
    return (1 - exp_m2z) / (1 + exp_m2z)


def sigmoid(Z):
    return 1 / (1 + T.exp(-Z))


def ReLU(Z):
    return T.switch(Z < 0, 0, Z)


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
