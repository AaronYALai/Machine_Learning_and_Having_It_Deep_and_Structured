# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-15 01:00:07
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-06 23:12:35

from unittest import TestCase
from DNN.run_DNN import run_model
from RNN_LSTM.run_RNN import run_RNN_model


class Test_running(TestCase):

    def test_DNN(self):
        run_model('train.data', 'train.label', 'test.data',
                  base_dir='./Data/')

    def test_RNN(self):
        run_RNN_model('train.data', 'train.label', 'ytrain_prob.npy',
                      'test.data', 'ytest_prob.npy', base_dir='./Data/',
                      epoch=10)
