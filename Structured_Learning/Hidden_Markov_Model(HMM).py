import numpy as np
import time
from collections import defaultdict

start_time = time.clock()
#The map from labels to integers(To compute y_hat) 
lab_index = dict([['aa', 0], ['ae', 1], ['ah', 2], ['ao', 3], ['aw', 4], ['ax', 5], ['ay', 6], ['b', 7], ['ch', 8], 
               ['cl', 9], ['d', 10], ['dh', 11], ['dx', 12], ['eh', 13], ['el', 14], ['en', 15], ['epi', 16], 
               ['er', 17], ['ey', 18], ['f', 19], ['g', 20], ['hh', 21], ['ih', 22], ['ix', 23], ['iy', 24], 
               ['jh', 25], ['k', 26], ['l', 27], ['m', 28], ['ng', 29], ['n', 30], ['ow', 31], ['oy', 32], ['p', 33], 
               ['r', 34], ['sh', 35], ['sil', 36], ['s', 37], ['th', 38], ['t', 39], ['uh', 40], ['uw', 41], ['vcl', 42],
               ['v', 43], ['w', 44], ['y', 45], ['zh', 46], ['z', 47]])

Y = open('train.lab')
Transition = np.zeros((48,48))
phone_prob = np.zeros((48,))
ID = ''; From = ''
for line in Y:
    line = line.strip().split(',')
    SeqID = '_'.join(line[0].split('_')[:2])
    To = line[1]
    phone_prob[lab_index[To]] += 1
    if SeqID == ID:
        Transition[lab_index[From]][lab_index[To]] += 1
    From = line[1]
    ID = SeqID
Y.close()
phone_prob = phone_prob/np.sum(phone_prob)
Trans_prob = (Transition.T/np.sum(Transition,axis=1)).T

def make_vector_sequence(Data,TID):
    """Concatenate vectors into single sequence of an utterance"""
    Dat = defaultdict(list)
    Number = {}; n = 0; old = ''
    for i in range(len(Data)):
        ID = '_'.join(TID[i].split('_')[:2])
        Dat[ID].append(Data[i])
        if ID != old:
            Number[n] = ID
            n += 1
            old = ID
    return Dat, Number

Mapp = open('48_39.map')
mapp = Mapp.readlines()
Mapp.close()

def Map_label(n):
    """Return label an integer(0-47) corresponds to"""
    for key,val in lab_index.items():
        if val == n:
            return key
    print("Worng!")

mapping = {}
for line in mapp:
    line = line.strip().split('\t')
    mapping[line[0]] = line[1]

M = open('48_idx_chr.map_b')
mapp = M.readlines()
M.close()

Map = {}
for line in mapp:
    line = line.strip().split(' ')
    word,_ = line[0].split('\t')
    Map[word] = line[-1]

print("Data loaded, using %f seconds"%(time.clock()-start_time))

print("Start predicting...")

Test = np.load('prob.out.npz')
Test = Test[Test.files[0]]
TID = np.load('TestID.npz')
TestID = TID[TID.files[0]][0]

TestD , Test_Num = make_vector_sequence(Test,TestID)

Seq_Pre = []
for z in range(len(Test_Num)):
    start = time.clock()
    Seq = TestD[Test_Num[z]]
    prob = np.ones((48,))/48
    predict = defaultdict(list)

    for l in Seq:
        for k in range(3):
            P = prob*(l/phone_prob)/9
            for i in range(48):
                L = P*Trans_prob.T[i]
                predict[i].append(mapping[Map_label(np.argmax(L))])
                prob[i] = np.max(L)
            if max(prob) >= 10**20:
                prob = prob/10**15

    Pre = []; now = -1; count = 0
    for n in predict[np.argmax(prob)]:
        if n != now:
            if count >= 9:
                if len(Pre)==0 or Pre[-1]!=now:
                    Pre.append(now)
            now = n
            count = 1
        else:
            count += 1
    if count >= 3:
        Pre.append(now)

    S = ''
    for i in Pre:
        S = S + Map[i]
    Seq_Pre.append(S[1:-1])

import csv
csvfile = open('XHMM.csv','w',newline='')
write = csv.writer(csvfile, delimiter=',')
write.writerow(['id','phone_sequence'])

for i in range(len(Test_Num)):
    write.writerow([Test_Num[i],Seq_Pre[i]])
    
print("All Done. Using %.4f seconds"%(time.clock()-start_time))
