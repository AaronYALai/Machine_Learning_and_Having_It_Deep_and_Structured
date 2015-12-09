import numpy as np
import time
from collections import defaultdict
from collections import Counter

start_time = time.clock()
#The map from labels to integers
lab_index = dict([['aa', 0], ['ae', 1], ['ah', 2], ['ao', 3], ['aw', 4], ['ax', 5], ['ay', 6], ['b', 7], ['ch', 8], 
               ['cl', 9], ['d', 10], ['dh', 11], ['dx', 12], ['eh', 13], ['el', 14], ['en', 15], ['epi', 16], 
               ['er', 17], ['ey', 18], ['f', 19], ['g', 20], ['hh', 21], ['ih', 22], ['ix', 23], ['iy', 24], 
               ['jh', 25], ['k', 26], ['l', 27], ['m', 28], ['ng', 29], ['n', 30], ['ow', 31], ['oy', 32], ['p', 33], 
               ['r', 34], ['sh', 35], ['sil', 36], ['s', 37], ['th', 38], ['t', 39], ['uh', 40], ['uw', 41], ['vcl', 42],
               ['v', 43], ['w', 44], ['y', 45], ['zh', 46], ['z', 47]])

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

def Map_label(n):
    """Return label an integer(0-47) corresponds to"""
    for key,val in lab_index.items():
        if val == n:
            return key
    print("Worng!")

mapp = open('48_39.map').readlines()
mapping = {}
for line in mapp:
    line = line.strip().split('\t')
    mapping[line[0]] = line[1]

Mapp = open('48_idx_chr.map_b').readlines()
Map = {}
for line in Mapp:
    line = line.strip().split(' ')
    word,_ = line[0].split('\t')
    Map[word] = line[-1]

file = open('train.lab')
TrainLab = {}; ID = ''; lab = []
TrainY = {}; seq = ''
for line in file:
    line = line.strip().split(',')
    SeqID = '_'.join(line[0].split('_')[:2])
    idx = Map[mapping[line[1]]]
    
    if SeqID != ID:
        TrainLab[ID] = lab
        lab = []
        TrainY[ID] = seq[2:-3]
        seq = ''
        
    lab.append(line[1])
    if len(seq)==0 or idx!=seq[-2]:
        seq = seq+idx+' '
    ID = SeqID
    
TrainLab[ID] = lab
TrainY[ID] = seq[2:-3]
file.close()

Train = np.load('prob_t.npz')
Train = Train[Train.files[0]]
TrID = np.load('TrainID.npz')
TrainID = TrID[TrID.files[0]][0]

TrainX , Train_Num = make_vector_sequence(Train,TrainID)

def Transition_proba(index):
    '''Calculate the transition probability matrix'''
    Transition = np.zeros((48,48))
    for ind in index:
        To = ''; From = ''; c = 0
        for cha in TrainLab[Train_Num[ind]]:
            To = cha
            c = c+1 if From == To else 0
            if len(From) != 0:
                Transition[lab_index[From]][lab_index[To]] += 1
            From = cha
    Transition = np.array(list(map(lambda x: np.log(x+2),Transition)))
    Trans_prob = (Transition.T/np.sum(Transition,axis=1)).T
    return Trans_prob

def toidx(seq,c):
    '''Transform the phoneme sequences into required format'''
    out = []; now = -1; count = 0
    for n in seq:
        if n != now:
            if count >= c:
                if len(out)==0 or out[-1]!=now:
                    out.append(now)
            now = n
            count = 1
        else:
            count += 1
    if count >= c:
        out.append(now)

    Str = ''
    for i in out:
        Str = Str + Map[i]
    return Str[1:-1]

def Voting(predicts):
    '''Blend the models uniformly'''
    result = []
    for i in range(len(predicts[0])):
        row = []
        for j in range(len(predicts[0][i])):
            l = []
            for k in range(len(predicts)):
                l.append(predicts[k][i][j])
            row.append(Counter(l).most_common()[0][0])
        result.append(row)
    return result

def edit(r,h):
    """edit distance"""
    r = r.split(); h = h.split()
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
                if i == 0:
                        d[0][j] = j
                elif j == 0:
                        d[i][0] = i

    for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                    if r[i-1] == h[j-1]:
                        d[i][j] = d[i-1][j-1]
                    else:
                        substitution = d[i-1][j-1] + 1
                        insertion    = d[i][j-1] + 1
                        deletion     = d[i-1][j] + 1
                        d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)]

def score(seq,ind,c):
    """Calculate the edit distance on validation set"""
    out = []; now = -1; count = 0
    for n in seq:
        if n != now:
            if count >= c:
                if len(out)==0 or out[-1]!=now:
                    out.append(now)
            now = n
            count = 1
        else:
            count += 1
    if count >= c:
        out.append(now)

    Str = ''
    for i in out:
        Str = Str + Map[i] + ' '
    return edit(TrainY[Train_Num[ind]],Str[2:-3])

print("Data loaded, using %f seconds"%(time.clock()-start_time))

print("Start predicting...")

Test = np.load('LSTM48-48.npz')
Test = Test[Test.files[0]]
TID = np.load('TestID.npz')
TestID = TID[TID.files[0]][0]

TestD , Test_Num = make_vector_sequence(Test,TestID)
Predict = {}
Valid = {}
Valid_ind = np.random.choice(len(Train_Num),len(Test_Num),replace=False)

duration = 3
for w in range(1): #change this when doing bagging
    Seq_Pre = []
    Val_Pre = []
    Trans_prob = Transition_proba(list(range(3696))) #np.random.choice(3696,3696)
    for z in range(len(Test_Num)): 
        Seq = TestD[Test_Num[z]]
        VSeq = TrainX[Train_Num[Valid_ind[z]]]
        prob = np.ones((48,))/48
        Vprob = np.ones((48,))/48
        predict = defaultdict(list)
        Vpredict = defaultdict(list)
        for l in Seq:
            P = prob*(l**duration)*Trans_prob.T 
            prob = np.max(P,axis=1)
            Arg = np.argmax(P,axis=1)
            for i in range(48):
                predict[i].append(mapping[Map_label(Arg[i])])

            if min(prob) <= 10**(-20): #Prevent probabilities from underflowing
                prob = prob*(10**15)

        for l in VSeq:
            P = Vprob*(l**duration)*Trans_prob.T #l = emission prob.
            Vprob = np.max(P,axis=1)
            Arg = np.argmax(P,axis=1)
            for i in range(48):
                Vpredict[i].append(mapping[Map_label(Arg[i])])

            if min(Vprob) <= 10**(-20):
                Vprob = Vprob*(10**15)

        Seq_Pre.append(predict[np.argmax(prob)])
        Val_Pre.append(Vpredict[np.argmax(Vprob)])
    Predict[w] = Seq_Pre
    Valid[w] = Val_Pre

Valid_seq = Valid[0]
Val = np.mean([score(Valid_seq[ind],val,3) for ind,val in enumerate(Valid_ind)])
print("The model has edit distance %.4f on the validation set."%(Val))

output = []
for p in Predict[0]:  #Voting(Predict)
    output.append(toidx(p,3))

import csv
csvfile = open('HMM_LSTM.csv','w',newline='')
write = csv.writer(csvfile, delimiter=',')
write.writerow(['id','phone_sequence'])

for i in range(len(Test_Num)):
    write.writerow([Test_Num[i],output[i]])

csvfile.close()    
print("All Done. Using %.4f seconds"%(time.clock()-start_time))

