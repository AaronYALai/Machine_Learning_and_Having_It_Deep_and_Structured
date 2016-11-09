# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-06 23:56:38
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-09 15:11:55

import numpy as np
import theano as th
import theano.tensor as T
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))    # noqa

from datetime import datetime
from shortcuts import load_data, load_label, make_data, make_y, load_str_map,\
                      validate, validate_editdist, test_predict
from activation import tanh, sigmoid, softmax
from optimize import sgd, momentum, NAG, RMSProp
from LSTM_utils import initialize_LSTM


def set_step(W_peephole, W_cell):
    U, Ui, Uf, Uo = W_peephole
    Vi, Vf, Vo = W_cell

    def step(z_t, zi_t, zf_t, zo_t, c_tm1, h_tm1):
        # new information
        Z_t = tanh(z_t + T.dot(h_tm1, U))

        # input gate
        Zi_t = sigmoid(zi_t + T.dot(h_tm1, Ui) + T.dot(c_tm1, Vi))

        # forget gate
        Zf_t = sigmoid(zf_t + T.dot(h_tm1, Uf) + T.dot(c_tm1, Vf))

        # new plus old/unforgetten memory
        c_t = Z_t * Zi_t + c_tm1 * Zf_t

        # output gate
        Zo_t = sigmoid(zo_t + T.dot(h_tm1, Uo) + T.dot(c_t, Vo))

        # output information
        h_t = tanh(c_t) * Zo_t

        return c_t, h_t

    return step


def construct_LSTM(n_input, n_output, n_hid_layers=2, archi=36, lr=1e-3,
                   update_by='NAG', batchsize=1, scale=0.01,
                   scale_b=0.001, clip_thres=1.0):
    """
    Initialize and construct the bidirectional Long Short-term Memory (LSTM)
    Update the LSTM using minibatch and RMSProp
    archi: number of neurons of each hidden layer
    """
    x_seq = T.fmatrix()
    y_hat = T.fmatrix()
    minibatch = T.scalar()

    # choose the optimization function
    optimiz_func = {
        'sgd': sgd,
        'momentum': momentum,
        'NAG': NAG,
        'RMSProp': RMSProp,
    }
    update_func = optimiz_func[update_by]

    # initialize the LSTM
    print('Start initializing LSTM...')
    init = initialize_LSTM(n_input, n_output, archi, n_hid_layers,
                           scale, scale_b, clip_thres)
    param_Ws, param_bs, auxis, caches, a_0, h_0, parameters = init

    # ############ bidirectional Long Short-term Memory ###############

    # #### Hidden layers ######
    for l in range(n_hid_layers):
        # computing gates
        if l == 0:
            a_seq = x_seq
            W, Wi, Wf, Wo = param_Ws[0][l][:-1]
            b, bi, bf, bo = param_bs[0][l]
            z_seq = T.dot(a_seq, W) + b.dimshuffle('x', 0)
            zi_seq = T.dot(a_seq, Wi) + bi.dimshuffle('x', 0)
            zf_seq = T.dot(a_seq, Wf) + bf.dimshuffle('x', 0)
            zo_seq = T.dot(a_seq, Wo) + bo.dimshuffle('x', 0)

            zf_seq, zif_seq, zff_seq, zof_seq = z_seq, zi_seq, zf_seq, zo_seq
            zb_seq, zib_seq, zfb_seq, zob_seq = z_seq, zi_seq, zf_seq, zo_seq
        else:
            # forward gates
            W_f, Wi_f, Wf_f, Wo_f = param_Ws[1][l - 1]
            b_f, bi_f, bf_f, bo_f = param_bs[1][l - 1]
            zf_seq = T.dot(a_seq, W_f) + b_f.dimshuffle('x', 0)
            zif_seq = T.dot(a_seq, Wi_f) + bi_f.dimshuffle('x', 0)
            zff_seq = T.dot(a_seq, Wf_f) + bf_f.dimshuffle('x', 0)
            zof_seq = T.dot(a_seq, Wo_f) + bo_f.dimshuffle('x', 0)

            # backward gates
            W_b, Wi_b, Wf_b, Wo_b = param_Ws[2][l - 1]
            b_b, bi_b, bf_b, bo_b = param_bs[2][l - 1]
            zb_seq = T.dot(a_seq, W_b) + b_b.dimshuffle('x', 0)
            zib_seq = T.dot(a_seq, Wi_b) + bi_b.dimshuffle('x', 0)
            zfb_seq = T.dot(a_seq, Wf_b) + bf_b.dimshuffle('x', 0)
            zob_seq = T.dot(a_seq, Wo_b) + bo_b.dimshuffle('x', 0)

        # computing cells
        step = set_step(param_Ws[3][l], param_Ws[4][l])

        # Forward direction
        seqs = [zf_seq, zif_seq, zff_seq, zof_seq]
        [cf_seq, hf_seq], _ = th.scan(step, sequences=seqs,
                                      outputs_info=[a_0, h_0],
                                      truncate_gradient=-1)

        # Backward direction
        seqs = [zb_seq[::-1], zib_seq[::-1], zfb_seq[::-1], zob_seq[::-1]]
        [cb_seq, hb_seq], _ = th.scan(step, sequences=seqs,
                                      outputs_info=[a_0, h_0],
                                      truncate_gradient=-1)

        a_seq = T.concatenate([hf_seq, hb_seq[::-1]], axis=1)

    # #### End of Hidden layers ######
    y_seq = softmax(T.dot(a_seq, param_Ws[0][0][-1]))
    forward = th.function(inputs=[x_seq], outputs=y_seq)

    cost = T.sum((y_seq - y_hat)**2) + minibatch * 0
    valid = th.function(inputs=[x_seq, y_hat, minibatch], outputs=cost)
    grads = T.grad(cost, parameters, disconnected_inputs='ignore')
    forward_grad = th.function([x_seq, y_hat, minibatch], grads)

    # ############ end of construction ###############

    updates = update_func(parameters, grads, lr, minibatch,
                          batchsize, auxis, caches)
    lstm_train = th.function(inputs=[x_seq, y_hat, minibatch],
                             outputs=cost, updates=updates)

    return forward, valid, lstm_train, forward_grad


def train_LSTM(trainX, train_label, forward, valid, lstm_train, forward_grad,
               n_output, int_str_map, batchsize, epoch=10, valid_ratio=0.2,
               print_every=20):
    """train the deep LSTM neural network"""
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
            costs += lstm_train(X_seq, trainY[speakers[num]], minibat_ind)
            n_instance += X_seq.shape[0]
            train_cost.append(costs / n_instance)

            # validation set
            if ind % print_every == (print_every - 1):
                v_cost = validate(trainX, trainY, valid_speakers, valid, None)
                valid_cost.append(v_cost)

                print('\tNow: %d; costs (train): %.4f ; costs (valid): %.4f' %
                      (j + 1, train_cost[-1], valid_cost[-1]))

                val_dist = validate_editdist(trainX, trainY, valid_speakers,
                                             forward, None, int_str_map)
                valid_dists.append(val_dist)
                print("\tEdit distance (valid): %.4f\n" % val_dist)

            # minibatch indicator plus 1
            minibat_ind = (minibat_ind + 1) % batchsize

    return train_cost, valid_cost, valid_dists


def run_LSTM_model(train_file, train_labfile, train_probfile, test_file=None,
                   test_probfile=None, neurons=36, n_hiddenlayer=2, lr=1e-3,
                   update_by='NAG', batchsize=1, epoch=10, valid_ratio=0.1,
                   n_input=48, n_output=48, save_prob=False,
                   base_dir='../Data/'):
    """Run the bidirectional deep Long Short-Term Memory network"""

    print("Start")
    st = datetime.now()

    data = load_data(base_dir + train_file)
    label_data, label_map = load_label(base_dir + train_labfile)
    int_str_map = load_str_map(label_map, base_dir)
    trainX, train_label = make_data(data, base_dir+train_probfile, label_data)
    print('Done loading data, using %s.' % str(datetime.now() - st))

    lstm = construct_LSTM(n_input, n_output, n_hiddenlayer, neurons, lr,
                          update_by, batchsize)
    forward, valid, lstm_train, forward_grad = lstm
    print('Done constructing the recurrent nueral network.')
    print('Using %s.\n' % str(datetime.now() - st))

    print('Start training LSTM...')
    train_LSTM(trainX, train_label, forward, valid, lstm_train, forward_grad,
               n_output, int_str_map, batchsize, epoch, valid_ratio)
    print('Done training, using %s.' % str(datetime.now() - st))

    if test_file and test_probfile:
        print('\nPredicting on test set...')
        test_predict(test_file, test_probfile, int_str_map, forward,
                     None, base_dir=base_dir, save_prob=save_prob,
                     prob_filename='LSTM_testprob')

    if save_prob:
        speakers = sorted(trainX.keys())
        probs = [forward(trainX[speaker]) for speaker in speakers]
        np.save('LSTM_trainprob', [probs, speakers])

    print("Done, Using %s." % str(datetime.now() - st))


def main():
    run_LSTM_model('train.data', 'train.label', 'ytrain_prob.npy', 'test.data',
                   'ytest_prob.npy', neurons=36, n_hiddenlayer=2, lr=1e-4,
                   update_by='NAG', batchsize=1, epoch=40, save_prob=True)


if __name__ == '__main__':
    main()
