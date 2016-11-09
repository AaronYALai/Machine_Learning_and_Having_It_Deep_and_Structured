# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-09 16:02:20
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-09 19:06:57

import numpy as np
import pandas as pd
import gc


def load_label(filename):
    """load label data"""
    label_data = pd.read_csv(filename, header=None, index_col=0)
    label_map = {}
    for ind, lab in enumerate(np.unique(label_data.values)):
        label_map[lab] = ind

    label_data = label_data.applymap(lambda x: label_map[x])
    gc.collect()

    return label_data, label_map


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
