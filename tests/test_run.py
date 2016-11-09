# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-15 01:00:07
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-09 22:29:45

from unittest import TestCase
from DNN.run_DNN import run_model
from RNN_LSTM.run_RNN import run_RNN_model
from RNN_LSTM.run_LSTM import run_LSTM_model
from HMM_topRNN.run_HMM import run_HMM


class Test_running(TestCase):

    def test_DNN(self):
        run_model('train.data', 'train.label', 'test.data',
                  base_dir='./Data/', save_prob=True, epoch=3)

    def test_RNN(self):
        run_RNN_model('train.data', 'train.label', 'ytrain_prob.npy',
                      'test.data', 'ytest_prob.npy', base_dir='./Data/',
                      epoch=3)

        run_RNN_model('train.data', 'train.label', 'ytrain_prob.npy',
                      'test.data', 'ytest_prob.npy', base_dir='./Data/',
                      acti_func='tanh', update_by='NAG', epoch=3)

        run_RNN_model('train.data', 'train.label', 'ytrain_prob.npy',
                      'test.data', 'ytest_prob.npy', base_dir='./Data/',
                      acti_func='sigmoid', update_by='momentum', epoch=3)

    def test_LSTM(self):
        run_LSTM_model('train.data', 'train.label', 'ytrain_prob.npy',
                       'test.data', 'ytest_prob.npy', base_dir='./Data/',
                       epoch=3, lr=1e-5)

    def test_HMM(self):
        run_HMM('RNN_trainprob.npy', 'train.label', 'RNN_testprob.npy',
                duration=3, blending=True, n_bag=10, valid_ratio=0.1,
                base_dir='./Data/')
