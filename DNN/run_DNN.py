# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-11 18:46:54
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-03 15:17:57
# flag: THEANO_FLAGS='floatX=float32'

import numpy as np
import pandas as pd
import theano as th
import theano.tensor as T
import gc
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))    # noqa

from datetime import datetime
from utils import load_data, load_label, initialize_NNet, maxout, \
                  softmax, update, gen_y_hat, accuracy


def construct_DNN(n_input, n_output, n_hid_layers=2, archi=128,
                  lr=1e-2, batchsize=40, dropout_rate=0.2, moment=0.95):
    """
    Initialize and construct the deep neural netweok with dropout
    update the DNN using momentum and minibatch
    archi: number of neurons of each hidden layer
    """
    # decide dropout or not, no dropout: stop_dropout > 1.05
    x = T.fmatrix()
    y_hat = T.fmatrix()
    stop_dropout = T.scalar()

    # initialize parameters
    Ws, bs, cache_Ws, cache_bs = initialize_NNet(n_input, n_output,
                                                 archi, n_hid_layers)

    # ############ construct the neural network ###############
    Zs = []
    As = []

    # input layer
    Zs.append(T.dot(x, Ws[0]) + bs[0].dimshuffle('x', 0))
    As.append(maxout(Zs[0], stop_dropout, archi, dropout_rate) / stop_dropout)

    # hidden layers
    for i in range(n_hid_layers):
        Zs.append(T.dot(As[i], Ws[i + 1]) + bs[i + 1].dimshuffle('x', 0))
        act_out = maxout(Zs[i + 1], stop_dropout, archi, dropout_rate)
        As.append(act_out / stop_dropout)

    # output layer
    z_out = T.dot(As[n_hid_layers], Ws[n_hid_layers + 1])
    Zs.append(z_out + bs[n_hid_layers + 1].dimshuffle('x', 0))
    y = softmax(Zs[-1] / stop_dropout)

    # ############ construct the neural network ###############

    forward = th.function([x, stop_dropout], y)
    parameters = Ws + bs
    moment_cache = cache_Ws + cache_bs

    # objective is the binary crossentropy
    Cost = ((-T.log((y * y_hat).sum(axis=1))).sum()) / batchsize

    # calculate gradients
    grads = T.grad(Cost, parameters, disconnected_inputs='ignore')

    # update parameters using momentum
    update_func = update(parameters, grads, moment_cache, lr, moment)
    gradient_update = th.function(inputs=[x, y_hat, stop_dropout],
                                  updates=update_func, outputs=Cost)

    return gradient_update, forward


def train_model(N, epoch, batchsize, gradient_update, feed_forward,
                data, label_data, n_output):
    train_start = datetime.now()
    obj_history = []
    valid_accu = []
    cache = {}

    for j in range(epoch):
        indexes = np.random.permutation(N - 8)
        objective = 0

        # train the model
        for i in range(int(N / batchsize)):
            if i % 1000 == 0:
                gc.collect()

            # make the minibatch data
            use_inds = indexes[i * batchsize:(i + 1) * batchsize] + 4
            batch_X = [data.iloc[(ind - 4):(ind + 5)].values.ravel()
                       for ind in use_inds]
            batch_Y = [gen_y_hat(ind, n_output, data, label_data, cache)
                       for ind in use_inds]
            # update the model
            objective += gradient_update(batch_X, batch_Y, 1)

        obj_history.append(objective / int(N / batchsize))
        print('\tepoch: %d; obj: %.4f' % (j + 1, obj_history[-1]))

        # validation set
        valid_accu.append(accuracy(N, data.shape[0], data, feed_forward,
                                   n_output, label_data, cache))

        print("\tCost: %.4f; , %.4f seconds used.\n" %
              (obj_history[-1],
               (datetime.now() - train_start).total_seconds()))
        # early stop
        if (valid_accu[0] != valid_accu[-1]):
            if valid_accu[-2] * 0.98 > valid_accu[-1]:
                print("Validation accuracy starts decreasing, stop training")
                break

    return obj_history, valid_accu, cache


def test_predict(test_file, label_map, forward, base_dir, save_prob=False,
                 filename='test_predict.csv'):
    print("Start predicting...")

    test_data = load_data(test_file)
    test_X = []
    test_N = len(test_data)
    # generate test input data
    for i in range(test_N):
        if i < 4:
            sils = np.zeros((4 - i) * test_data.shape[1])
            dat = test_data.iloc[:(i + 5)].values.ravel()
            test_X.append(np.concatenate((sils, dat)))

        elif i > (test_N - 5):
            dat = test_data.iloc[(i - 4):].values.ravel()
            sils = np.zeros((5 - test_N + i) * test_data.shape[1])
            test_X.append(np.concatenate((dat, sils)))

        else:
            test_X.append(test_data.iloc[(i - 4):(i + 5)].values.ravel())

    y_test_pred = forward(test_X, np.float32(1.25))

    if save_prob:
        np.save('ytest_prob', y_test_pred)

    # find the mapping from int to phoneme
    phoneme_map = {}
    pmap = pd.read_csv(base_dir + '48_39.map', sep='\t', header=None)
    for p1, p2 in pmap.values:
        phoneme_map[p1] = p2

    int_phoneme_map = {}
    for key, val in label_map.items():
        int_phoneme_map[val] = phoneme_map[key]

    test_phon = [int_phoneme_map[np.argmax(y_vec)] for y_vec in y_test_pred]
    data = {'Prediction': test_phon, 'Id': test_data.index.values}
    test_df = pd.DataFrame(data=data)
    test_df.to_csv(filename, index=None)


def run_model(train_file, train_labfile, test_file=None, valid_ratio=0.05,
              batchsize=40, epoch=5, base_dir='./Data/', save_prob=False):
    print("Start")
    st = datetime.now()

    data = load_data(base_dir + train_file)
    label_data, label_map = load_label(base_dir + train_labfile)

    # window size = 9, output = 48 phonemes
    n_input = data.shape[1] * 9
    n_output = 48
    N = int(data.shape[0] * (1 - valid_ratio))

    print("Done loading data. Start constructing the model...")
    gradient_update, feed_forward = construct_DNN(n_input, n_output, archi=36)

    print("Finish constructing the model. Start Training...")
    result = train_model(N, epoch, batchsize, gradient_update,
                         feed_forward, data, label_data, n_output)
    obj_history, valid_accu, cache = result

    # train accuracy
    train_accu = accuracy(0, N, data, feed_forward, n_output,
                          label_data, cache)
    print("Training Accuracy: %.4f %%" % (100 * train_accu))

    # validation
    valid_accu = accuracy(N, data.shape[0], data, feed_forward,
                          n_output, label_data, cache)
    print("Validation Accuracy: %.4f %%" % (100 * valid_accu))

    if save_prob:
        accuracy(0, data.shape[0], data, feed_forward, n_output,
                 label_data, cache, save_pred=True, save_name='ytrain_prob')

    if test_file:
        test_predict(base_dir + test_file, label_map, feed_forward,
                     base_dir, save_prob=save_prob)

    print("Done, Using %s." % str(datetime.now() - st))


def main():
    run_model('train.data', 'train.label', 'test.data', save_prob=True)


if __name__ == '__main__':
    main()
