# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-12 16:25:45
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-06 17:33:05

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
        X[speaker] = (df.iloc[speaker_indexes].values).astype('float32')
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
                   clip_thres=1.0):
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
        b_memory.append(th.shared(random_number([archi], scale_b)))

        if i == (n_hid_layers - 1):
            continue

        W_out_forward.append(th.shared(random_number([2*archi, archi], scale)))
        W_out_backward.append(th.shared(random_number([2*archi, archi], scale)))
        b_out_forward.append(th.shared(random_number([archi], scale_b)))
        b_out_backward.append(th.shared(random_number([archi], scale_b)))

    # output layer
    W_in_out.append(th.shared(random_number([2 * archi, n_output], scale)))
    b_in_out.append(th.shared(random_number([n_output], scale_b)))

    param_Ws = [W_in_out, W_out_forward, W_out_backward, W_memory]
    param_bs = [b_in_out, b_out_forward, b_out_backward, b_memory]

    # help to do advanced optimization (ex. NAG, RMSProp)
    aux_Ws = []
    aux_bs = []

    # help to do mini-batch update (to store gradients)
    cache_Ws = []
    cache_bs = []

    parameters = []
    for i in range(4):
        aux_W = []
        aux_b = []
        cache_W = []
        cache_b = []

        parameters += param_Ws[i]
        parameters += param_bs[i]

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

    # concatenate all auxilary and cache parameters
    auxis = []
    caches = []
    for i in range(4):
        auxis += aux_Ws[i]
        auxis += aux_bs[i]

        caches += cache_Ws[i]
        caches += cache_bs[i]

    return param_Ws, param_bs, auxis, caches, a_0, parameters


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


def make_y(lab, n_output):
    y = np.zeros(n_output)
    y[lab] = 1
    return y


def validate(trainX, trainY, valid_speakers, valid, dropout_rate):
    objective = 0
    n_instance = 0
    stop = 1.0 / (1 - dropout_rate)

    for speaker in valid_speakers:
        objective += valid(trainX[speaker], trainY[speaker], 0, stop)
        n_instance += trainX[speaker].shape[0]

    return objective / n_instance


def edit_dist(seq1, seq2):
    """edit distance"""
    seq1 = seq1.split()
    seq2 = seq2.split()

    d = np.zeros((len(seq1) + 1) * (len(seq2) + 1), dtype=np.uint8)
    d = d.reshape((len(seq1) + 1, len(seq2) + 1))

    for i in range(len(seq1) + 1):
        for j in range(len(seq2) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i - 1] == seq2[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(seq1)][len(seq2)]


def validate_editdist(trainX, trainY, valid_speakers, forward, dropout_rate, int_phoneme_map):
    """Calculate the average edit distance on validation set"""
    stop = 1.0 / (1 - dropout_rate)

    valid_seq = []
    valid_yhat_seq = []
    for speaker in valid_speakers:
        phoneme_seq = ''
        last = ''

        for pred in forward(trainX[speaker], stop):
            phoneme = int_phoneme_map[np.argmax(pred)]
            if last != phoneme:
                phoneme_seq = phoneme_seq + phoneme + ' '
            last = phoneme

        yhat_seq = ' '.join([int_phoneme_map[i] for i in trainY[speaker].ravel()])

        valid_seq.append(phoneme_seq.strip())
        valid_yhat_seq.append(yhat_seq)

    valid_dist = np.mean([edit_dist(valid_seq[i], valid_yhat_seq[i]) for i in range(len(valid_seq))])

    return valid_dist


def load_phoneme_map(label_map, base_dir='./'):
    # find the mapping from int to phoneme
    phoneme_map = {}
    pmap = pd.read_csv(base_dir + '48_39.map', sep='\t', header=None)
    for p1, p2 in pmap.values:
        phoneme_map[p1] = p2

    int_phoneme_map = {}
    for key, val in label_map.items():
        int_phoneme_map[val] = phoneme_map[key]

    return int_phoneme_map


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
