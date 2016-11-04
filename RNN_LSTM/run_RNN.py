# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-03 11:40:23
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-03 16:33:06

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
                  softmax, update, gen_y_hat, accuracy, make_data


data = load_data('Data/train.data')
label_data, label_map = load_label('Data/train.label')
trainX, train_label = make_data(data, 'Data/ytrain_prob.npy', label_data)



import pdb;pdb.set_trace()

