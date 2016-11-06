# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-03 11:40:23
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-06 17:38:57

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
                  softmax, update, gen_y_hat, accuracy, make_data, make_y, validate, validate_editdist, load_phoneme_map

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

n_input = 48
n_output = 48
batchsize = 1

param_Ws, param_bs, auxis, caches, a_0, parameters = initialize_RNN(n_input, n_output)

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
        a_seq = x_seq
        z_seq = T.dot(a_seq, param_Ws[0][l]) + param_bs[0][l].dimshuffle('x', 0)
        Z_forward_seqs.append(z_seq)
        Z_backward_seqs.append(z_seq)
    else:
        Z_forward_seqs.append(T.dot(a_seq, param_Ws[1][l - 1]) + param_bs[1][l - 1].dimshuffle('x',0))
        Z_backward_seqs.append(T.dot(a_seq, param_Ws[2][l - 1]) + param_bs[2][l - 1].dimshuffle('x',0))

    step = set_step(param_Ws[3], param_bs[3], l)
    [af_seq, ab_seq], _ = th.scan(step, sequences=[Z_forward_seqs[l], Z_backward_seqs[l][::-1]], 
                                  outputs_info=[a_0,a_0],
                                  truncate_gradient=-1)

    A_forward_seqs.append(af_seq)
    A_backward_seqs.append(ab_seq)
    a_out = T.concatenate([af_seq, ab_seq[::-1]], axis=1)
    a_seq = ifelse(T.lt(stop_dropout, 1.05),
                    (a_out * srng.binomial(size=T.shape(a_out), p=(1 - dropout_rate))).astype('float32'),
                    a_out) / stop_dropout

y_pre = T.dot(a_seq, param_Ws[0][1]) + param_bs[0][1].dimshuffle('x',0)
y_seq = softmax(y_pre)
forward = th.function(inputs=[x_seq, stop_dropout], outputs=y_seq)

cost = T.sum((y_seq - y_hat)**2) + minibatch * 0
valid = th.function(inputs=[x_seq, y_hat, minibatch, stop_dropout], outputs=cost)

grads = T.grad(cost, parameters, disconnected_inputs='ignore')

def update_by_sgd(parameters, grads, lr, minibatch, batchsize, auxis, caches):
    updates = []
    update_batch = ifelse(T.lt(minibatch, batchsize - 1), 0, 1)

    for ix in range(len(grads)):
        # update parameters if reaching batchsize
        move = -(lr / batchsize) * (caches[ix] + grads[ix])
        updates.append((parameters[ix], parameters[ix] + move * update_batch))
        updates.append((caches[ix], (caches[ix] + grads[ix]) * (1 - update_batch)))

    return updates


def update_by_momentum(parameters, grads, lr, minibatch, batchsize, momentum, caches, moment=0.95):
    """theano update, optimized by Momentum"""
    updates = []
    update_batch = ifelse(T.lt(minibatch, batchsize - 1), 0, 1)

    for ix in range(len(grads)):
        direction = moment * momentum[ix] - (lr / batchsize) * (grads[ix] + caches[ix])

        # update parameters if reaching batchsize
        updates.append((parameters[ix], parameters[ix] + direction * update_batch))
        # remember the move if updating parameters
        updates.append((momentum[ix], momentum[ix] * (1 - update_batch) + direction * update_batch))
        # accumulate gradients if not reaching batchsize
        updates.append((caches[ix], (caches[ix] + grads[ix]) * (1 - update_batch)))

    return updates


def update_by_NAG(parameters, grads, lr, minibatch, batchsize, real_pos, caches, moment=0.95):
    """theano update, optimized by NAG"""
    updates = []
    update_batch = ifelse(T.lt(minibatch, batchsize - 1), 0, 1)

    for ix in range(len(grads)):
        move = -(lr / batchsize) * (caches[ix] + grads[ix])
        real_position = parameters[ix] + move
        spy_position = real_position + moment * (real_position - real_pos[ix])

        # update parameters to spy position if reaching batchsize
        updates.append((parameters[ix], spy_position * update_batch + parameters[ix] * (1 - update_batch)))
        # remember the real position if moved parameters
        updates.append((real_pos[ix], real_position * update_batch + real_pos[ix] * (1 - update_batch)))
        # accumulate gradients if not reaching batchsize
        updates.append((caches[ix], (caches[ix] + grads[ix]) * (1 - update_batch)))

    return updates

#(parameters, grads, lr, minibatch, batchsize, real_pos, caches, const=0.001)
def update_by_RMSProp(para,grad,ind,Sigma_square,Temp):
    """theano update, optimized by RMSProp"""
    updates = []
    update_batch = ifelse(T.lt(minibatch, batchsize - 1), 0, 1)

    for ix in range(len(grads)):
        grad[ix] = T.clip(grad[ix],-1,1)
        gradient = (grad[ix]+Temp[ix])/b
        Factor = Sigma_square[ix]*alpha+(1-alpha)*(gradient**2)
        direction = -(learing_rate)*gradient/(T.sqrt(Factor)+0.001)
        updates.append((para[ix], (para[ix]+direction)*off_on+para[ix]*(1-off_on)))
        updates.append((Sigma_square[ix], Factor*off_on+Sigma_square[ix]*(1-off_on)))
        updates.append((Temp[ix], (Temp[ix]+grad[ix])*(1-off_on)))
    return updates


import pdb;pdb.set_trace()



lr = 0.00003


update_func = update_by_NAG(parameters, grads, lr, minibatch, batchsize, auxis, caches)
rnn_train = th.function(inputs=[x_seq, y_hat, minibatch, stop_dropout], outputs=cost,
                        updates=update_func)


speakers = sorted(trainX.keys())

trainY = {}
for speaker in speakers:
    y = [make_y(lab, n_output) for lab in train_label[speaker].ravel()]
    trainY[speaker] = np.array(y).astype('float32')

epoch = 5
print_every = 5
valid_ratio = 0.2

valid_n = round(len(speakers) * valid_ratio)
rand_speakers = np.random.permutation(speakers)
valid_speakers = rand_speakers[:valid_n]
train_speakers = rand_speakers[valid_n:]

valid_dists = []
train_cost = []
valid_cost = []

int_phoneme_map = load_phoneme_map(label_map)
for j in range(epoch):
    objective = 0 
    n_instance = 0
    minibat_ind = 0

    indexes = np.random.permutation(len(train_speakers))
    for ind, num in enumerate(indexes):
        X_seq = trainX[speakers[num]]
        objective += rnn_train(X_seq, trainY[speakers[num]], minibat_ind, 1)
        n_instance += X_seq.shape[0]
        train_cost.append(objective / n_instance)

        if ind % print_every == (print_every - 1):
            v_cost = validate(trainX, trainY, valid_speakers, valid, dropout_rate)
            valid_cost.append(v_cost)

            print('\ttraining costs: %f ; validation Cost: %f' %
                  (train_cost[-1], valid_cost[-1]))

            val_editdist = validate_editdist(trainX, trainY, valid_speakers, forward,
                                    dropout_rate, int_phoneme_map)
            valid_dists.append(val_editdist)
            print("\tEdit distance on validation set: %f" % val_editdist)

        minibat_ind = (minibat_ind + 1) % batchsize

import pdb;pdb.set_trace()

