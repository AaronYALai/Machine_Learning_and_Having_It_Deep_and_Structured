# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-12 16:25:45
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-06 22:51:19

import numpy as np
import pandas as pd
import gc


def load_data(filename, nrows=None, normalize=True):
    """load data from file, first column as index, dtype=float32"""
    ind = pd.read_csv(filename, sep=' ', header=None, index_col=0, nrows=5)
    dtype_dict = {c: np.float32 for c in ind.columns}
    data = pd.read_csv(filename, sep=' ', header=None, index_col=0,
                       dtype=dtype_dict, nrows=nrows)
    # normalize
    if normalize:
        data = (data - data.mean()) / data.std()
        gc.collect()

    return data


def load_label(filename):
    """load label data"""
    label_data = pd.read_csv(filename, header=None, index_col=0)
    label_map = {}
    for ind, lab in enumerate(np.unique(label_data.values)):
        label_map[lab] = ind

    label_data = label_data.applymap(lambda x: label_map[x])
    gc.collect()

    return label_data, label_map


def make_data(data, prob_file, label_data=None):
    """transform data into one sequence for each speaker"""
    prob_data = np.load(prob_file)
    df = pd.DataFrame(data=prob_data, index=data.index)
    speakers = list(set(['_'.join(name.split('_')[:2]) for name in df.index]))

    X = {}
    labels = {}
    for speaker in speakers:
        speaker_indexes = df.index.str.startswith(speaker)
        X[speaker] = (df.iloc[speaker_indexes].values).astype('float32')
        if label_data is not None:
            labels[speaker] = label_data.iloc[speaker_indexes].values

    return X, labels


def make_y(lab, n_output):
    """make y vector"""
    y = np.zeros(n_output)
    y[lab] = 1
    return y


def validate(trainX, trainY, valid_speakers, valid, dropout_rate):
    """Calculate the cost value on validation set"""
    objective = 0
    n_instance = 0
    stop = 1.0 / (1 - dropout_rate)

    for speaker in valid_speakers:
        objective += valid(trainX[speaker], trainY[speaker], 0, stop)
        n_instance += trainX[speaker].shape[0]

    return objective / n_instance


def load_str_map(label_map, base_dir='../Data/'):
    """find the mapping from int to phoneme"""
    phoneme_map = {}
    phone_str_map = {}
    pmap = pd.read_csv(base_dir + '48_39.map', sep='\t', header=None)
    str_map = pd.read_csv(base_dir + '48_idx_chr.map',
                          header=None, delim_whitespace=True)

    for p1, p2 in pmap.values:
        phoneme_map[p1] = p2

    for s1, s2, s3 in str_map.values:
        phone_str_map[s1] = s3

    int_str_map = {}
    for key, val in label_map.items():
        int_str_map[val] = phone_str_map[phoneme_map[key]]

    return int_str_map


def sanity_check(seq, sep=' '):
    """Sanity Check function to correct unreasonable predictions"""
    seq = seq.split()

    for i in range(1, len(seq) - 1):
        # front == behind != me
        if seq[i - 1] == seq[i + 1] and seq[i] != seq[i - 1]:
            seq[i] = seq[i - 1]
        # me, front, behind are different
        elif seq[i] != seq[i + 1] and seq[i] != seq[i - 1]:
            seq[i] = seq[i - 1]

    return sep.join(seq)


def edit_dist(seq1, seq2):
    """edit distance"""
    seq1 = seq1.split()
    seq2 = seq2.split()

    d = np.zeros((len(seq1) + 1) * (len(seq2) + 1), dtype=np.uint8)
    d = d.reshape((len(seq1) + 1, len(seq2) + 1))

    for i in range(len(seq1) + 1):
        for j in range(len(seq2) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i - 1] == seq2[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(seq1)][len(seq2)]


def validate_editdist(trainX, trainY, valid_speakers, forward,
                      dropout_rate, int_str_map):
    """Calculate the average edit distance on validation set"""
    stop = 1.0 / (1 - dropout_rate)

    valid_seq = []
    valid_y_seq = []
    for speaker in valid_speakers:
        ypred = forward(trainX[speaker], stop)
        pred_seq = ' '.join([int_str_map[np.argmax(pred)] for pred in ypred])
        pred_seq = sanity_check(pred_seq)

        phoneme_seq = ''
        now = ''
        for p in pred_seq.split():
            if p != now:
                phoneme_seq += (p + ' ')
                now = p

        yhat_seq = [int_str_map[np.argmax(l)] for l in trainY[speaker]]
        yhat = []
        y_now = ''

        for y in yhat_seq:
            if y != y_now:
                yhat.append(y)
                y_now = y

        yhat = ' '.join(yhat)

        valid_seq.append(phoneme_seq.strip())
        valid_y_seq.append(yhat)

    leng = len(valid_seq)
    dists = [edit_dist(valid_seq[i], valid_y_seq[i]) for i in range(leng)]
    valid_dist = np.mean(dists)

    return valid_dist


def test_predict(testfile, testprob_file, int_str_map, forward, dropout_rate,
                 filename='test.csv', base_dir='../Data/'):
    """predict on test set and output the file"""
    test_data = load_data(base_dir + testfile)
    testX, _ = make_data(test_data, base_dir + testprob_file)

    test_speakers = list(testX.keys())
    stop = 1.0 / (1 - dropout_rate)

    test_speakers = []
    now_speak = ''
    for s in test_data.index:
        speaker = '_'.join(s.split('_')[:2])
        if speaker != now_speak:
            test_speakers.append(speaker)
            now_speak = speaker

    test_seq = []
    for speaker in test_speakers:
        pred_seq = forward(testX[speaker], stop)
        pred_seq = [int_str_map[np.argmax(pred)] for pred in pred_seq]
        pred_seq = ' '.join(pred_seq)
        pred_seq = sanity_check(pred_seq)

        seq = ''
        now = ''
        for p in pred_seq.split():
            if p != now:
                seq += p
                now = p

        test_seq.append(seq)

    test_pred = {'id': test_speakers, 'phone_sequence': test_seq}
    test_df = pd.DataFrame(data=test_pred)
    test_df.to_csv(filename, index=None)
