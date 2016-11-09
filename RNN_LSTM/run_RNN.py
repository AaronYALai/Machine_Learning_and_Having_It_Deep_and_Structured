# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-03 11:40:23
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-09 15:28:40

import numpy as np
import theano as th
import theano.tensor as T
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))    # noqa

from datetime import datetime
from shortcuts import load_data, load_label, make_data, make_y, load_str_map,\
                      validate, validate_editdist, test_predict
from activation import tanh, sigmoid, ReLU, softmax
from optimize import sgd, momentum, NAG, RMSProp
from RNN_utils import initialize_RNN


from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams


def set_step(W_memory, b_memory, lay_j, acti_func='ReLU'):
    functions = {
        'ReLU': ReLU,
        'sigmoid': sigmoid,
        'tanh': tanh,
    }
    activ = functions[acti_func]

    def step(zf_t, zb_t, af_tm1, ab_tm1):
        af_t = activ(zf_t + T.dot(af_tm1, W_memory[lay_j]) + b_memory[lay_j])
        ab_t = activ(zb_t + T.dot(ab_tm1, W_memory[lay_j]) + b_memory[lay_j])
        return af_t, ab_t

    return step


def construct_RNN(n_input, n_output, n_hid_layers=2, archi=128, lr=1e-3,
                  acti_func='ReLU', update_by='RMSProp', dropout_rate=0.2,
                  batchsize=1, scale=0.033, scale_b=0.001, clip_thres=10.0,
                  seed=42):
    """
    Initialize and construct the bidirectional deep RNN with dropout
    Update the RNN using minibatch and RMSProp
    archi: number of neurons of each hidden layer
    """
    x_seq = T.fmatrix()
    y_hat = T.fmatrix()
    minibatch = T.scalar()
    stop_dropout = T.scalar()

    # choose the optimization function
    optimiz_func = {
        'sgd': sgd,
        'momentum': momentum,
        'NAG': NAG,
        'RMSProp': RMSProp,
    }
    update_func = optimiz_func[update_by]

    # initialize the RNN
    print('Start initializing RNN...')
    init = initialize_RNN(n_input, n_output, archi, n_hid_layers,
                          scale, scale_b, clip_thres)
    param_Ws, param_bs, auxis, caches, a_0, parameters = init

    # ############ bidirectional recurrent neural network ###############
    srng = RandomStreams(seed=seed)

    # #### Hidden layers ######
    for l in range(n_hid_layers):
        if l == 0:
            a_seq = x_seq
            z_seq = T.dot(a_seq, param_Ws[0][l])
            z_seq += param_bs[0][l].dimshuffle('x', 0)
            zf_seq = z_seq
            zb_seq = z_seq
        else:
            zf_seq = T.dot(a_seq, param_Ws[1][l - 1])
            zf_seq += param_bs[1][l - 1].dimshuffle('x', 0)
            zb_seq = T.dot(a_seq, param_Ws[2][l - 1])
            zb_seq += param_bs[2][l - 1].dimshuffle('x', 0)

        step = set_step(param_Ws[3], param_bs[3], l, acti_func)
        [af_seq, ab_seq], _ = th.scan(step, sequences=[zf_seq, zb_seq[::-1]],
                                      outputs_info=[a_0, a_0],
                                      truncate_gradient=-1)

        a_out = T.concatenate([af_seq, ab_seq[::-1]], axis=1)
        dropping = srng.binomial(size=T.shape(a_out),
                                 p=(1 - dropout_rate))
        a_seq = ifelse(T.lt(stop_dropout, 1.05),
                       (a_out * dropping).astype('float32'), a_out)
        a_seq /= stop_dropout

    # #### End of Hidden layers ######

    y_pre = T.dot(a_seq, param_Ws[0][1]) + param_bs[0][1].dimshuffle('x', 0)
    y_seq = softmax(y_pre)
    forward = th.function(inputs=[x_seq, stop_dropout], outputs=y_seq)

    cost = T.sum((y_seq - y_hat)**2) + minibatch * 0
    valid = th.function(inputs=[x_seq, y_hat, minibatch, stop_dropout],
                        outputs=cost)
    grads = T.grad(cost, parameters, disconnected_inputs='ignore')

    # ############ end of construction ###############

    updates = update_func(parameters, grads, lr, minibatch,
                          batchsize, auxis, caches)
    rnn_train = th.function(inputs=[x_seq, y_hat, minibatch, stop_dropout],
                            outputs=cost, updates=updates)

    return forward, valid, rnn_train


def train_RNN(trainX, train_label, forward, valid, rnn_train, n_output,
              int_str_map, dropout_rate, batchsize, epoch=10, valid_ratio=0.2,
              print_every=20):
    """train the deep recurrent neural network"""
    speakers = sorted(trainX.keys())

    # making training y sequence
    trainY = {}
    for speaker in speakers:
        y = [make_y(lab, n_output) for lab in train_label[speaker].ravel()]
        trainY[speaker] = np.array(y).astype('float32')

    # split the validation set
    valid_n = round(len(speakers) * valid_ratio)
    rand_speakers = np.random.permutation(speakers)
    valid_speakers = rand_speakers[:valid_n]
    train_speakers = rand_speakers[valid_n:]

    valid_dists = []
    train_cost = []
    valid_cost = []

    # training process
    for j in range(epoch):
        costs = 0
        n_instance = 0
        minibat_ind = 0

        # random shuffle the order
        indexes = np.random.permutation(len(train_speakers))
        for ind, num in enumerate(indexes):
            X_seq = trainX[speakers[num]]
            costs += rnn_train(X_seq, trainY[speakers[num]], minibat_ind, 1)
            n_instance += X_seq.shape[0]
            train_cost.append(costs / n_instance)

            # validation set
            if ind % print_every == (print_every - 1):
                v_cost = validate(trainX, trainY, valid_speakers,
                                  valid, dropout_rate)
                valid_cost.append(v_cost)

                print('\tNow: %d; costs (train): %.4f ; costs (valid): %.4f' %
                      (j + 1, train_cost[-1], valid_cost[-1]))

                val_dist = validate_editdist(trainX, trainY, valid_speakers,
                                             forward, dropout_rate,
                                             int_str_map)
                valid_dists.append(val_dist)
                print("\tEdit distance (valid): %.4f\n" % val_dist)

            # minibatch indicator plus 1
            minibat_ind = (minibat_ind + 1) % batchsize

    return train_cost, valid_cost, valid_dists


def run_RNN_model(train_file, train_labfile, train_probfile, test_file=None,
                  test_probfile=None, neurons=36, n_hiddenlayer=2, lr=1e-3,
                  acti_func='ReLU', update_by='RMSProp', dropout_rate=0.2,
                  batchsize=1, epoch=10, valid_ratio=0.1, n_input=48,
                  n_output=48, base_dir='../Data/', save_prob=False):
    """Run the bidirectional deep recurrent neural network with droput"""

    print("Start")
    st = datetime.now()

    data = load_data(base_dir + train_file)
    label_data, label_map = load_label(base_dir + train_labfile)
    int_str_map = load_str_map(label_map, base_dir)
    trainX, train_label = make_data(data, base_dir+train_probfile, label_data)
    print('Done loading data, using %s.' % str(datetime.now() - st))

    rnn = construct_RNN(n_input, n_output, n_hiddenlayer, neurons, lr,
                        acti_func, update_by, dropout_rate, batchsize)
    forward, valid, rnn_train = rnn
    print('Done constructing the recurrent nueral network.\n')

    print('Start training RNN...')
    train_RNN(trainX, train_label, forward, valid, rnn_train, n_output,
              int_str_map, dropout_rate, batchsize, epoch, valid_ratio)
    print('Done training, using %s.' % str(datetime.now() - st))

    if test_file and test_probfile:
        print('\nPredicting on test set...')
        test_predict(test_file, test_probfile, int_str_map, forward,
                     dropout_rate, base_dir=base_dir, save_prob=save_prob,
                     prob_filename='RNN_testprob')

    if save_prob:
        speakers = sorted(trainX.keys())
        stop = 1 / (1 - dropout_rate)
        probs = [forward(trainX[speaker], stop) for speaker in speakers]
        np.save('RNN_trainprob', [probs, speakers])

    print("Done, Using %s." % str(datetime.now() - st))


def main():
    run_RNN_model('train.data', 'train.label', 'ytrain_prob.npy', 'test.data',
                  'ytest_prob.npy', neurons=128, n_hiddenlayer=2, lr=1e-3,
                  acti_func='ReLU', update_by='RMSProp', dropout_rate=0.2,
                  batchsize=1, epoch=100, save_prob=True)


if __name__ == '__main__':
    main()
