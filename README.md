# ML-Deep-and-Structured

[![Build Status](https://travis-ci.org/AaronYALai/Machine_Learning_and_Having_It_Deep_and_Structured.svg?branch=master)](https://travis-ci.org/AaronYALai/Machine_Learning_and_Having_It_Deep_and_Structured)
[![Coverage Status](https://coveralls.io/repos/github/AaronYALai/Machine_Learning_and_Having_It_Deep_and_Structured/badge.svg?branch=master)](https://coveralls.io/github/AaronYALai/Machine_Learning_and_Having_It_Deep_and_Structured?branch=master)

Implementation about 
- Deep Neural Network: Maxout(generalization of ReLU) activation function, Softmax output layer, Cross entropy cost function, Dropout technique, and Momentum optimizations to train a DNN to classify phonemes.  
- Recurrent Neural Network: Use Bi-directional RNN and LSTM(peephole) with RMSProp/NAG optimization and Dropout technique to improve the results of DNN just trained. 
- Structured learining: On top of results of RNN/LSTM, applying Hidden Markov Model to model the phone transition probabilities and further improves the performance of RNN/LSTM.

The performance is measured by Levenshtein distance(a.k.a. Edit distance).

Given each time frame to phone label probabilities learned from LSTM(Long Short-term Memory), utilize Hidden Markov Model to model the phone transition probabilities, see the whole utterance as one training data, and output phone label sequence. The performance is measured by Levenshtein distance(a.k.a. Edit distance).
