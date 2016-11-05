# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-03 11:40:23
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-06 00:22:13

import numpy as np
import pandas as pd
import theano as th
import theano.tensor as T
import gc
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))    # noqa

from datetime import datetime
from utils import load_data, load_label, initialize_RNN, tanh, sigmoid, ReLU,\
                  softmax, update, gen_y_hat, accuracy, make_data

from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams


def set_step(W_memory, b_memory, layer_j, acti_func='ReLU'):
    functions = {
        'ReLU': ReLU,
        'sigmoid': sigmoid,
        'tanh': tanh,
    }
    activ = functions[acti_func]

    def step(zf_t, zb_t, af_tm1, ab_tm1):
        af_t = activ(zf_t + T.dot(af_tm1, W_memory[layer_j]) + b_memory[layer_j])
        ab_t = activ(zb_t + T.dot(ab_tm1, W_memory[layer_j]) + b_memory[layer_j])
        return af_t, ab_t

    return step

data = load_data('Data/train.data')
label_data, label_map = load_label('Data/train.label')
trainX, train_label = make_data(data, 'Data/ytrain_prob.npy', label_data)

x_seq = T.fmatrix()
y_hat = T.fmatrix()
minibatch = T.scalar()
stop_dropout = T.scalar()

param_Ws, param_bs, aux_Ws, aux_bs, cache_Ws, cache_bs, a_0, parameters = initialize_RNN(48, 48)

# construct RNN
n_hid_layers = 2
seed = 42
dropout_rate = 0.2
srng = RandomStreams(seed=seed)

Z_forward_seqs = []
Z_backward_seqs = []
A_forward_seqs = []
A_backward_seqs = []

# #### Hidden layers ######
for l in range(n_hid_layers):
    if l == 0:
        z_seq = T.dot(x_seq, param_Ws[0][l]) + param_bs[0][l].dimshuffle('x', 0)
        Z_forward_seqs.append(z_seq)
        Z_backward_seqs.append(z_seq)
    else:
        Z_forward_seqs.append(T.dot(a_seq, param_Ws[1][l - 1]) + param_bs[1][l - 1].dimshuffle('x',0))
        Z_backward_seqs.append(T.dot(a_seq, param_Ws[2][l - 1]) + param_bs[2][l - 1].dimshuffle('x',0))

    step = set_step(param_Ws[3], param_bs[3], l)
    [af_seq, ab_seq], _ = th.scan(step, sequences = [Z_forward_seqs[l], Z_backward_seqs[l][::-1]], 
                                  outputs_info = [a_0,a_0],
                                  truncate_gradient=-1)

    A_forward_seqs.append(af_seq)
    A_backward_seqs.append(ab_seq)
    a_out = T.concatenate([af_seq, ab_seq[::-1]], axis=1)
    a_seq = ifelse(T.lt(stop_dropout, 1.05),
                    (a_out * srng.binomial(size=T.shape(a_out), p=(1 - dropout_rate))).astype('float32'),
                    a_out) / stop_dropout

y_pre = T.dot(a_seq, param_Ws[0][1]) + param_bs[0][1].dimshuffle('x',0)
y_seq = softmax(y_pre)
forword = th.function(inputs=[x_seq, stop_dropout], outputs=y_seq)

cost = T.sum((y_seq - y_hat)**2) + minibatch * 0
valid = th.function(inputs=[x_seq, y_hat, minibatch, stop_dropout], outputs=cost)

grads = T.grad(cost, parameters, disconnected_inputs='ignore')

def Update(parameters, grads, lr):
    return [(parameters[i], parameters[i] - lr * grads[i]) for i in range(len(parameters))]

lr = 1e-2
rnn_train = th.function(inputs=[x_seq, y_hat, minibatch, stop_dropout], outputs=cost,
                        updates=Update(parameters, grads, lr))


import pdb;pdb.set_trace()

