# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-15 01:00:07
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-06 18:00:28

from unittest import TestCase
from DNN.run_DNN import run_model

class Test_running(TestCase):

    def test_DNN(self):
        run_model('train.data', 'train.label', 'test.data',
                  base_dir='./Data/')
