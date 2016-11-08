# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-07 02:10:33
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-09 00:27:23


import numpy as np
import theano as th


def initialize_LSTM(n_input, n_output, archi=48, n_hid_layers=2,
                    scale=0.01, scale_b=0.001, clip_thres=0.3):
    """initialize the LSTM paramters, archi: hidden layer neurons"""
    W_in_out = []
    W_gate_forward = []
    W_gate_backward = []
    W_cell = []
    W_peephole = []

    b_in_out = []
    b_gate_forward = []
    b_gate_backward = []

    # initial cell and output h
    a_0 = th.shared(random_number([archi], 0))
    h_0 = th.shared(random_number([archi], 0))

    # hidden layers
    for i in range(n_hid_layers):
        # initilize peephole parameters
        U = th.shared(random_number([archi, archi], scale))
        Ui = th.shared(random_number([archi, archi], scale))
        Uf = th.shared(identity_mat(archi, scale))
        Uo = th.shared(random_number([archi, archi], scale))
        W_peephole.append([U, Ui, Uf, Uo])

        # initialize memory cell paramters
        Vi = th.shared(random_number([archi, archi], scale))
        Vf = th.shared(identity_mat(archi, scale))
        Vo = th.shared(random_number([archi, archi], scale))
        W_cell.append([Vi, Vf, Vo])

        # input layer
        if i == 0:
            Ws, bs = init_gate_params([n_input, archi], [archi],
                                      scale, scale_b)

            W_output = th.shared(random_number([2 * archi, n_output], scale))
            W_in_out.append(Ws + [W_output])
            b_in_out.append(bs)

        else:
            Ws_forw, bs_forw = init_gate_params([2 * archi, archi], [archi],
                                                scale, scale_b)
            W_gate_forward.append(Ws_forw)
            b_gate_forward.append(bs_forw)

            Ws_back, bs_back = init_gate_params([2 * archi, archi], [archi],
                                                scale, scale_b)
            W_gate_backward.append(Ws_back)
            b_gate_backward.append(bs_back)

    param_Ws = [W_in_out, W_gate_forward, W_gate_backward, W_peephole, W_cell]
    param_bs = [b_in_out, b_gate_forward, b_gate_backward]

    parameters = [w for Ws in param_Ws for W in Ws for w in W]
    parameters += [b for bs in param_bs for bb in bs for b in bb]

    # help to do advanced optimization (ex. NAG, RMSProp)
    auxis = [th.shared(zero_number(p.get_value().shape)) for p in parameters]

    # help to do mini-batch update (to store gradients)
    caches = [th.shared(zero_number(p.get_value().shape)) for p in parameters]

    # set the restricted numerical range for gradient values
    for i in range(len(param_Ws)):
        for j in range(len(param_Ws[i])):
            for k in range(len(param_Ws[i][j])):
                param_Ws[i][j][k] = th.gradient.grad_clip(param_Ws[i][j][k],
                                                          -clip_thres,
                                                          clip_thres)

    for i in range(len(param_bs)):
        for j in range(len(param_bs[i])):
            for k in range(len(param_bs[k])):
                param_bs[i][j][k] = th.gradient.grad_clip(param_bs[i][j][k],
                                                          -clip_thres,
                                                          clip_thres)

    return param_Ws, param_bs, auxis, caches, a_0, h_0, parameters


def init_gate_params(W_shape, b_shape, scale, scale_b):
    W = th.shared(random_number(W_shape, scale))
    Wi = th.shared(random_number(W_shape, scale))
    Wf = th.shared(random_number(W_shape, scale) + np.float32(scale / 2))
    Wo = th.shared(random_number(W_shape, scale))

    b = th.shared(random_number(b_shape, scale_b))
    bi = th.shared(random_number(b_shape, scale_b))
    bf = th.shared(random_number(b_shape, scale_b))
    bo = th.shared(random_number(b_shape, scale_b))

    return [W, Wi, Wf, Wo], [b, bi, bf, bo]


def random_number(shape, scale=1):
    return (scale * np.random.randn(*shape)).astype('float32')


def zero_number(shape):
    return np.zeros(shape).astype('float32')


def identity_mat(N, scale):
    return (scale * np.identity(N)).astype('float32')
