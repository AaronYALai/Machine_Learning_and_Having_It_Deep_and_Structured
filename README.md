Machine Learning and having it deep and structured
========

[![Build Status](https://travis-ci.org/AaronYALai/Machine_Learning_and_Having_It_Deep_and_Structured.svg?branch=master)](https://travis-ci.org/AaronYALai/Machine_Learning_and_Having_It_Deep_and_Structured)
[![Coverage Status](https://coveralls.io/repos/github/AaronYALai/Machine_Learning_and_Having_It_Deep_and_Structured/badge.svg?branch=master)](https://coveralls.io/github/AaronYALai/Machine_Learning_and_Having_It_Deep_and_Structured?branch=master)

About
--------

Implementations and homeworks of the course **Machine Learning and having it deep and structured** of National Taiwan University (offered by [**Hung-yi Lee**](http://speech.ee.ntu.edu.tw/~tlkagk/index.html):

- Constructed and trained variants of neural networks by [**Theano**](http://deeplearning.net/software/theano/)
- Attemped to solve the sequence labeling problem in speech recognition (phoneme labeling)
- Deep Neural Network (DNN) with dropout, maxout and momentum optimization
- Bidirectional Recurrent Neural Network (RNN) with dropout and RMSProp optimization
- Bidirectional Long-Short Term Memory (LSTM) with peephole and NAG optimization
- Hidden Markov Model (HMM) on top of RNN to improve the performance

Syllabus
--------

Neural Networks and Training:
- What is Machine Learning, Deep Learning and Structured Learning?
- Neural Network Basics | Backpropagation | Theano: DNN
- Tips for Training Deep Neural Network
- Neural Network with Memory | Theano: RNN
- Training Recurrent Neural Network
- Convolutional Neural Network (by Prof. Winston)

Structured Learning and Graphical Models:
- Introduction of Structured Learning | Structured Linear Model | Structured SVM
- Sequence Labeling Problem | Learning with Hidden Information
- Graphical Model, Gibbs Sampling

Extensions, New Applications and Trends:
- Markov Logic Network
- Deep Learning for Human Language Processing, Language Modeling
- Caffe | Deep Reinforcement Learning | Visual Question Answering
- Unsupervised Learning
- Attention-based Model

Content
--------

Deep Neural Network (DNN)[[kaggle](https://inclass.kaggle.com/c/104-1-mlds-hw1)]:
- Construct and train a deep neural network to classify pronunciation units (phonemes) in each time frame of a speech.
- Inputs: MFCC features
- Activation function: **Maxout** (generalization of ReLU, "learnable" activation function) 
- Output layer: Softmax
- Cost function: cross entropy
- Optimization: Momentum
- With **Dropout** technique


Bidirectional Recurrent Neural Network (RNN)[[kaggle](https://inclass.kaggle.com/c/104-1-mlds-hw2)]:
- Construct and train a bidirectional deep recurrent neural network to classify pronunciation units (phonemes) in each time frame of a speech.
- Inputs: prediction probabilities of each class from previous DNN
- Activation function: ReLU
- Output layer: Softmax
- Cost function: Mean Squared Error
- Optimization: Root Mean Square Propagation (RMSProp)
- With **Dropout** technique

Bidirectional Long-Short Term Memory (LSTM)[[kaggle](https://inclass.kaggle.com/c/104-1-mlds-hw2)]:
- Construct and train a bidirectional deep Long-Short Term Memory to classify pronunciation units (phonemes) in each time frame of a speech.
- Inputs: prediction probabilities of each class from previous DNN
- Optimization: Nesterov Accelerated Gradient (NAG)
- With **peephole** technique

Structure Learning (output phone label sequence)[[kaggle](https://inclass.kaggle.com/c/104-1-mlds-hw3)]:
- On top of results of RNN / LSTM, applies Hidden Markov Model (HMM) to model the phone transition probabilities and further improves the performance of RNN / LSTM on this sequence labeling problem.
- Inputs: the whole utterance as one training data
- Outputs: phone label sequence

The performance is measured by Levenshtein distance (a.k.a. Edit distance).

Usage
--------
Clone the repo and use the [virtualenv](http://www.virtualenv.org/):

    git clone https://github.com/AaronYALai/Machine_Learning_and_Having_It_Deep_and_Structured

    cd Machine_Learning_and_Having_It_Deep_and_Structured

    virtualenv venv

    source venv/bin/activate

Install all dependencies and run the model:

    pip install -r requirements.txt

    cd RNN_LSTM

    python run_RNN.py
