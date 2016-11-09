# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-09 15:54:45
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-09 22:35:03

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))    # noqa

from HMM_utils import load_label, load_str_map, sanity_check, edit_dist
from collections import defaultdict, Counter
from datetime import datetime


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


def HMM_predict(seq_probs, labels, speakers, n_phoneme, int_str_map,
                test_probs=None, duration=3, blending=False, n_bag=1):
    """generate a bag of prediction sequences for each speaker"""
    if not blending:
        n_bag = 1

    predict_bags = []
    test_bags = []
    for num in range(n_bag):
        predictions = []
        test_predicts = []

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

        # test set
        if test_probs is None:
            continue

        for test_seq in test_probs:
            test_score = np.ones((n_phoneme,)) / n_phoneme
            testpred_seq = defaultdict(list)

            for test_vec in test_seq:
                test_matrix = transition_prob * (test_vec**duration)
                test_matrix *= test_score
                test_score = np.max(test_matrix, axis=1)
                test_inds = np.argmax(test_matrix, axis=1)
                # normalize
                test_score /= test_score.sum()

                # compute the predicted phoneme with starting phoneme i
                for i in range(n_phoneme):
                    testpred_seq[i].append(int_str_map[test_inds[i]])

            # choose the sequence with the highest score
            test_predicts.append(testpred_seq[np.argmax(test_score)])

        test_bags.append(test_predicts)

    return predict_bags, test_bags


def voting(predict_bags):
    """voting of a bag of sequences to make the final sequence"""
    result = []
    for i in range(len(predict_bags[0])):
        bag_seqs = np.array([pred[i] for pred in predict_bags])
        seq = [Counter(l).most_common()[0][0] for l in bag_seqs.T]
        result.append(seq)

    return result


def output_seq(pred_seq, sep=''):
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
        label_result.append(output_seq(seq, sep=' '))

    return label_result


def run_HMM(train_probfile, train_labfile, test_probfile=None, n_phoneme=48,
            duration=3, blending=False, n_bag=10, valid_ratio=0.1,
            base_dir='../Data/'):
    print("Start")
    st = datetime.now()

    # loading data
    label_data, label_map = load_label(base_dir + train_labfile)
    train_probs, train_speakers = np.load(base_dir + train_probfile)
    int_str_map = load_str_map(label_map, base_dir)

    if test_probfile:
        test_probs, test_speakers = np.load(base_dir + test_probfile)
    else:
        test_probs = None

    print('Done loading data, using %s.\n' % str(datetime.now() - st))

    print('Start using HMM for predictions...')
    # computing label sequence for each speaker
    labels = {}
    for speaker in train_speakers:
        speaker_indexes = label_data.index.str.startswith(speaker)
        labels[speaker] = label_data.iloc[speaker_indexes].values

    # split into training and validation set
    n_speaker = len(train_speakers)
    rand_inds = np.random.permutation(n_speaker)
    valid_inds = rand_inds[:int(n_speaker * valid_ratio)]
    train_inds = rand_inds[int(n_speaker * valid_ratio):]

    # predict sequences using HMM with blending
    bags = HMM_predict(train_probs, labels, train_speakers[train_inds],
                       n_phoneme, int_str_map, test_probs, duration, blending,
                       n_bag)
    predict_bags, test_bags = bags
    predict_result = voting(predict_bags)

    if len(test_bags) > 0:
        test_predict = voting(test_bags)

    # transform to alphabet sequences and compute the edit distances
    predict_result = [output_seq(pred_seq, sep=' ')
                      for pred_seq in predict_result]
    label_result = make_label_seq(labels, train_speakers, int_str_map)
    print('Done predicting, using %s.' % str(datetime.now() - st))

    # evaluate training set
    train_predict = np.array(predict_result)[train_inds]
    train_lab = np.array(label_result)[train_inds]
    train_scores = [edit_dist(train_lab[i], train_predict[i])
                    for i in range(len(train_predict))]

    # evaluate validation set
    valid_predict = np.array(predict_result)[valid_inds]
    valid_lab = np.array(label_result)[valid_inds]
    valid_scores = [edit_dist(valid_lab[i], valid_predict[i])
                    for i in range(len(valid_predict))]

    print("\nEdit distance (train): %.4f" % np.mean(train_scores))
    print("Edit distance (valid): %.4f\n" % np.mean(valid_scores))

    # output predict file
    if test_probfile:
        test_predict_seqs = [output_seq(test_seq, sep='')
                             for test_seq in test_predict]
        test_pred = {'id': test_speakers, 'phone_sequence': test_predict_seqs}
        test_df = pd.DataFrame(data=test_pred)
        test_df.to_csv('HMM_testpredict.csv', index=None)

    print("Done, Using %s." % str(datetime.now() - st))


def main():
    run_HMM('RNN_trainprob.npy', 'train.label', 'RNN_testprob.npy',
            duration=3, blending=True, n_bag=100, valid_ratio=0.2,
            base_dir='../Data/')


if __name__ == '__main__':
    main()
