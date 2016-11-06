# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-06 21:06:57
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-06 21:10:01

import theano.tensor as T
import theano as th


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
