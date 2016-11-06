# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-06 20:50:39
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-06 22:26:54


import numpy as np
import theano as th


def initialize_RNN(n_input, n_output, archi=128, n_hid_layers=2,
                   scale=0.033, scale_b=0.001, clip_thres=3.0):
    """initialize the RNN paramters, archi: hidden layer neurons"""
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
        rand_w = random_number([2*archi, archi], scale)
        W_out_backward.append(th.shared(rand_w))
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


def random_number(shape, scale=1):
    return (scale * np.random.randn(*shape)).astype('float32')


def zero_number(shape):
    return np.zeros(shape).astype('float32')


def identity_mat(N, scale):
    return (scale * np.identity(N)).astype('float32')
