# ML-Deep-and-Structured
Implementation about 
- Deep Neural Network: Maxout(generalization of ReLU) activation function, Softmax output layer, Cross entropy cost function, Dropout technique, and Momentum optimizations to train a DNN to classify phonemes.  
- Recurrent Neural Network: Use Bi-directional RNN and LSTM(peephole) with RMSProp/NAG optimization and Dropout technique to improve the results of DNN just trained. 
- Structured learining: On top of results of RNN/LSTM, applying Hidden Markov Model to model the phone transition probabilities and further improves the performance of RNN/LSTM.

The performance is measured by Levenshtein distance(a.k.a. Edit distance).
