# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-09 15:54:45
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-09 19:06:36

import numpy as np

from HMM_utils import load_label, load_str_map, sanity_check, edit_dist
from collections import defaultdict, Counter


def make_transMat(labels, speakers, n_phoneme):
    """computing the transition matrix using label sequence at hand"""
    transition_prob = np.zeros((n_phoneme, n_phoneme))

    for speaker in speakers:
        previous = labels[speaker][0][0]
        for phoneme in labels[speaker][1:]:
            transition_prob[phoneme[0], previous] += 1
            previous = phoneme[0]

    transition_prob = np.log(transition_prob + 2)
    transition_prob /= transition_prob.sum(axis=0)

    return transition_prob


def HMM_predict(seq_probs, labels, speakers, n_phoneme, duration=3,
                blending=False, n_bag=1):
    """generate a bag of prediction sequences for each speaker"""
    if not blending:
        n_bag = 1

    predict_bags = []
    for num in range(n_bag):
        predictions = []
        # calculate transition prob

        if blending:
            bagspeakers = np.random.choice(speakers, len(speakers))
            transition_prob = make_transMat(labels, bagspeakers, n_phoneme)
        else:
            transition_prob = make_transMat(labels, speakers, n_phoneme)

        for seq in seq_probs: 
            prob_score = np.ones((n_phoneme,)) / n_phoneme
            predict_seq = defaultdict(list)

            for vec in seq:
                prob_matrix = prob_score * (vec**duration) * transition_prob
                prob_score = np.max(prob_matrix, axis=1)
                pred_inds = np.argmax(prob_matrix, axis=1)
                # normalize
                prob_score /= prob_score.sum()

                # compute the predicted phoneme with starting phoneme i
                for i in range(n_phoneme):
                    predict_seq[i].append(int_str_map[pred_inds[i]])

            # choose the sequence with the highest score
            predictions.append(predict_seq[np.argmax(prob_score)])

        predict_bags.append(predictions)

    return predict_bags


def voting(predict_bags):
    """voting of a bag of sequences to make the final sequence"""
    result = []
    for i in range(len(predict_bags[0])):
        bag_seqs = np.array([pred[i] for pred in predict_bags])
        seq = [Counter(l).most_common()[0][0] for l in bag_seqs.T]
        result.append(seq)

    return result


def output_phoneme_seq(pred_seq, sep=''):
    pred_seq = sanity_check(' '.join(pred_seq))

    phoneme_seq = ''
    now = ''
    for p in pred_seq.split():
        if p != now:
            phoneme_seq += (p + sep)
            now = p

    return phoneme_seq.strip()


def make_label_seq(labels, speakers, int_str_map):
    """transform the labels to str sequence"""
    label_result = []

    for speaker in speakers:
        seq = ' '.join([int_str_map[ind[0]] for ind in labels[speaker]])
        label_result.append(output_phoneme_seq(seq, sep=' '))

    return label_result



base_dir = '../Data/'
train_labfile = 'train.label'
train_probfile = 'RNN_trainprob.npy'
test_probfile = 'RNN_testprob.npy'

label_data, label_map = load_label(base_dir + train_labfile)
train_probs, train_speakers = np.load(base_dir + train_probfile)
test_probs, test_speakers = np.load(base_dir + test_probfile)


labels = {}
for speaker in train_speakers:
    speaker_indexes = label_data.index.str.startswith(speaker)
    labels[speaker] = label_data.iloc[speaker_indexes].values

int_str_map = load_str_map(label_map, base_dir)
n_phoneme = 48
selected_speakers = np.random.choice(train_speakers, 10)

# predition
duration = 3

n_bag = 1
blending = False

seq_probs = train_probs

predict_bags = HMM_predict(seq_probs, labels, train_speakers, n_phoneme, duration, blending=True, n_bag=5)

result = voting(predict_bags)

result = [output_phoneme_seq(pred_seq, sep=' ') for pred_seq in result]

label_result = make_label_seq(labels, train_speakers, int_str_map)

editdists = [edit_dist(label_result[i], result[i]) for i in range(len(label_result))]
import pdb;pdb.set_trace()

