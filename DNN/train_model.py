# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-11 18:46:54
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-10-13 18:55:03

import numpy as np
import theano as th
import theano.tensor as T
import gc
import time

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
    As.append(maxout(Zs[0], stop_dropout, 128, dropout_rate) / stop_dropout)

    # hidden layers
    for i in range(n_hid_layers):
        Zs.append(T.dot(As[i], Ws[i + 1]) + bs[i + 1].dimshuffle('x', 0))
        act_out = maxout(Zs[i + 1], stop_dropout, 128, dropout_rate)
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


def run_model(train_file, train_labfile, valid_ratio=0.05,
              batchsize=40, epoch=5):
    st = time.clock()

    print("Start")
    data = load_data(train_file, nrows=100000)
    label_data, label_map = load_label(train_labfile)

    # window size = 9, output = 48 phonemes
    n_input = data.shape[1] * 9
    n_output = 48
    N = int(data.shape[0] * (1 - valid_ratio))

    print("Done loading data. Start constructing the model...")
    gradient_update, forward = construct_DNN(n_input, n_output)

    print("Finish constructing the model. Start Training...")
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
        valid_accu.append(accuracy(N, data.shape[0], data, forward, n_output,
                                   label_data, cache))

        print("\tCost: %.4f; , %.4f seconds used." %
              (obj_history[-1], time.clock() - st))
        if (valid_accu[0] != valid_accu[-1]):
            if valid_accu[-2] * 0.98 > valid_accu[-1]:
                print("Validation accuracy starts decreasing, stop training")
                break

    train_accu = accuracy(0, N, data, forward, n_output, label_data, cache)
    print("Training Accuracy: %.4f %%" % (100 * train_accu))

    valid_accu = accuracy(N, data.shape[0], data, forward,
                          n_output, label_data, cache)

    print("Validation Accuracy: %.4f %%" % (100 * valid_accu))
    print("Done, Using %.4f seconds." % (time.clock() - st))


def main():
    run_model('train.ark', 'train.lab')


if __name__ == '__main__':
    main()
