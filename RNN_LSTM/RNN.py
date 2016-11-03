#Bi-directional RNN, Momentum, NAG, RMSProp, Dropout 
import theano as th
import theano.tensor as T
import numpy as np
import time
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams
from collections import defaultdict

start_time = time.clock()
Data = np.load('/Users/aaronlai/Desktop/G/Data/prob_t.npz') #DNN's output
Data = Data[Data.files[0]]
F = np.load('/Users/aaronlai/Desktop/G/Data/TrainID.npz')
TrainID = F[F.files[0]][0]

Label = {}
Y = open('../train.lab')
for line in Y:
    line = line.strip().split(',')
    Label[line[0]] = line[1]
Y.close()

#The map from labels to integers(To compute y_hat) 
lab_index = dict([['aa', 0], ['ae', 1], ['ah', 2], ['ao', 3], ['aw', 4], ['ax', 5], ['ay', 6], ['b', 7], ['ch', 8], 
               ['cl', 9], ['d', 10], ['dh', 11], ['dx', 12], ['eh', 13], ['el', 14], ['en', 15], ['epi', 16], 
               ['er', 17], ['ey', 18], ['f', 19], ['g', 20], ['hh', 21], ['ih', 22], ['ix', 23], ['iy', 24], 
               ['jh', 25], ['k', 26], ['l', 27], ['m', 28], ['ng', 29], ['n', 30], ['ow', 31], ['oy', 32], ['p', 33], 
               ['r', 34], ['sh', 35], ['sil', 36], ['s', 37], ['th', 38], ['t', 39], ['uh', 40], ['uw', 41], ['vcl', 42],
               ['v', 43], ['w', 44], ['y', 45], ['zh', 46], ['z', 47]])

def y_hat(i):
    """give the np array of y_hat"""
    l = np.zeros(48)
    l[lab_index[Label[TrainID[i]]]] = 1
    return l

def make_vector_sequence(Data,TID,have_label=True):
    """Concatenate vectors into single sequence of an utterance"""
    Dat = defaultdict(list); Y_hat = defaultdict(list)
    Number = {}; n = 0; old = ''
    for i in range(len(Data)):
        ID = '_'.join(TID[i].split('_')[:2])
        Dat[ID].append(Data[i])
        if have_label:
            Y_hat[ID].append(y_hat(i))
        if ID != old:
            Number[n] = ID
            n += 1
            old = ID
    return Dat, Y_hat, Number

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

TrainX, TrainY, Train_Num =  make_vector_sequence(Data,TrainID)
Order = np.random.permutation(len(Train_Num))

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

Yval_cache = {}
def Valid_Y():
    """Return Y_hat of the validation set"""
    try:
        return Yval_cache[0]
    except:
        Val_Y = []
        for i in range(3000,len(Train_Num)):
            yval = ''; last = ''
            for l in TrainY[Train_Num[Order[i]]]:
                phone = Map[mapping[Map_label(np.argmax(l))]]
                if last != phone:
                    yval = yval + phone + ' '
                last = phone
            Val_Y.append(yval[:-1])

        Yval_cache[0] = Val_Y
        return Val_Y

print("Data loaded, using %f seconds"%(time.clock()-start_time))
import pdb;pdb.set_trace()
#RNN structure
st = time.clock()
b = 1    #Batch size
learing_rate = 0.00003
srng = RandomStreams(seed=5432)

x_seq = T.fmatrix()
y_hat = T.fmatrix()
ind = T.scalar()    #Help to do minibatch
bud = T.scalar()    #Help to do Dropout 

cons = 0.001; a=0.0; s=0.01; neuron = 160
a_0 = th.shared(0*np.random.randn(neuron))
Wi = th.shared(s*np.random.randn(48,neuron))
bi = th.shared(cons*np.random.randn(neuron)-a)

Wh = th.shared(s*np.identity(neuron)-a)
Wof = th.shared(s*np.random.randn(2*neuron,neuron)-a)
Wob = th.shared(s*np.random.randn(2*neuron,neuron)-a)
bh = th.shared(cons*np.random.randn(neuron)-a)
bof = th.shared(cons*np.random.randn(neuron)-a)
bob = th.shared(cons*np.random.randn(neuron)-a)
"""
W2h = th.shared(s*np.identity(neuron)-a)
W2of = th.shared(s*np.random.randn(2*neuron,neuron)-a)
W2ob = th.shared(s*np.random.randn(2*neuron,neuron)-a)
b2h = th.shared(cons*np.random.randn(neuron)-a)
b2of = th.shared(cons*np.random.randn(neuron)-a)
b2ob = th.shared(cons*np.random.randn(neuron)-a)
"""
W3h = th.shared(s*np.identity(neuron)-a)
W3o = th.shared(s*np.random.randn(2*neuron,48)-a)
b3h = th.shared(cons*np.random.randn(neuron)-a)
b3o = th.shared(cons*np.random.randn(48)-a)

Auxiliary = []; Temp = []
parameters = [Wi,bi,Wh,Wof,Wob,bh,bof,bob,W3h,W3o,b3h,b3o]#W2h,W2of,W2ob,b2h,b2of,b2ob,
for param in parameters:
    Auxiliary.append(th.shared(np.zeros(param.get_value().shape)))
    Temp.append(th.shared(np.zeros(param.get_value().shape)))
    
c = 5
Wi = th.gradient.grad_clip(Wi,-c,c)
bi = th.gradient.grad_clip(bi,-c,c)
Wh = th.gradient.grad_clip(Wh,-c,c)
Wof = th.gradient.grad_clip(Wof,-c,c)
Wob = th.gradient.grad_clip(Wob,-c,c)
bh = th.gradient.grad_clip(bh,-c,c)
bof = th.gradient.grad_clip(bof,-c,c)
bob = th.gradient.grad_clip(bob,-c,c)
W3h = th.gradient.grad_clip(W3h,-c,c)
W3o = th.gradient.grad_clip(W3o,-c,c)
b3h = th.gradient.grad_clip(b3h,-c,c)
b3o = th.gradient.grad_clip(b3o,-c,c)  

def Update_Momentum(para,grad,ind,Momentum,Temp):
    """theano update, optimized by Momentum"""
    updates = []; off_on = ifelse(T.lt(ind,b-1),0,1)
    for ix in range(len(grad)):
        #gradient = T.clip(grad[ix],-1,1)
        direction = (0.95)*Momentum[ix] - (learing_rate/b)*(gradient+Temp[ix])
        updates.append((para[ix], para[ix]+direction*off_on))
        updates.append((Momentum[ix], Momentum[ix]*(1-off_on)+direction*off_on))
        updates.append((Temp[ix], (Temp[ix]+gradient)*(1-off_on)))
    return updates

def Update_NAG(para,grad,ind,Real,Temp):
    """theano update, optimized by NAG"""
    updates = []; off_on = ifelse(T.lt(ind,b-1),0,1)
    for ix in range(len(grad)):
        #grad[ix] = T.clip(grad[ix],-1,1)
        gradient = -(learing_rate/b)*(grad[ix]+Temp[ix])
        spy_position = (1+0.95)*(para[ix]+gradient)-0.95*Real[ix]
        updates.append((para[ix], (spy_position)*off_on+para[ix]*(1-off_on)))
        updates.append((Real[ix], (para[ix]+gradient)*off_on+Real[ix]*(1-off_on)))
        updates.append((Temp[ix], (Temp[ix]+grad[ix])*(1-off_on)))
    return updates

def Update_RMSProp(para,grad,ind,Sigma_square,Temp):
    """theano update, optimized by RMSProp"""
    updates = []; off_on = ifelse(T.lt(ind,b-1),0,1); alpha = 0.9
    for ix in range(len(grad)):
        grad[ix] = T.clip(grad[ix],-1,1)
        gradient = (grad[ix]+Temp[ix])/b
        Factor = Sigma_square[ix]*alpha+(1-alpha)*(gradient**2)
        direction = -(learing_rate)*gradient/(T.sqrt(Factor)+0.001)
        updates.append((para[ix], (para[ix]+direction)*off_on+para[ix]*(1-off_on)))
        updates.append((Sigma_square[ix], Factor*off_on+Sigma_square[ix]*(1-off_on)))
        updates.append((Temp[ix], (Temp[ix]+grad[ix])*(1-off_on)))
    return updates

def sigmoid(Z):
    return 1/(1+T.exp(-Z))

def ReLU(Z):
    return T.switch(Z<0,0,Z)

def softmax(Z):
    z = T.exp(Z)
    return (z.T/T.sum(z,axis=1)).T

def step(zf_t,zb_t,af_tm1,ab_tm1):
    af_t = ReLU( zf_t + T.dot(af_tm1,Wh) + bh )
    ab_t = ReLU( zb_t + T.dot(ab_tm1,Wh) + bh )
    return af_t, ab_t

def step2(zf_t,zb_t,af_tm1,ab_tm1):
    af_t = ReLU( zf_t + T.dot(af_tm1,W2h) + b2h )
    ab_t = ReLU( zb_t + T.dot(ab_tm1,W2h) + b2h )
    return af_t, ab_t

def step3(zf_t,zb_t,af_tm1,ab_tm1):
    af_t = ReLU( zf_t + T.dot(af_tm1,W3h) + b3h )
    ab_t = ReLU( zb_t + T.dot(ab_tm1,W3h) + b3h )
    return af_t, ab_t

##### Layer 1 ######
z_seq = T.dot(x_seq,Wi)+bi.dimshuffle('x',0)
[af_seq,ab_seq],_ = th.scan(step, sequences = [z_seq,z_seq[::-1]], 
                               outputs_info = [a_0,a_0],
                              truncate_gradient=-1)
a_out = T.concatenate([af_seq,ab_seq[::-1]],axis=1)
a_seq = ifelse(T.lt(bud,1.05), a_out*srng.binomial(size=T.shape(a_out),p=0.8),a_out)/bud

##### Layer 2 ######
z1_f = T.dot(a_seq,Wof)+bof.dimshuffle('x',0)
z1_b = T.dot(a_seq,Wob)+bob.dimshuffle('x',0)
[a2f_seq,a2b_seq],_ = th.scan(step3, sequences = [z1_f,z1_b[::-1]], 
                               outputs_info = [a_0,a_0],
                              truncate_gradient=-1)
"""
a2_seq = T.concatenate([a2f_seq,a2b_seq[::-1]],axis=1)

##### Layer 3 ######
z2_f = T.dot(a2_seq,W2of)+b2of.dimshuffle('x',0)
z2_b = T.dot(a2_seq,W2ob)+b2ob.dimshuffle('x',0)

[a3f_seq,a3b_seq],_ = th.scan(step3, sequences = [z2_f,z2_b[::-1]], 
                               outputs_info = [a_0,a_0],
                              truncate_gradient=-1)
"""

a3_out = T.concatenate([a2f_seq,a2b_seq[::-1]],axis=1)
a3_seq = ifelse(T.lt(bud,1.05), a3_out*srng.binomial(size=T.shape(a3_out),p=0.8),a3_out)/bud
y3_pre = T.dot(a3_seq,W3o)+b3o.dimshuffle('x',0)

y_seq = softmax(y3_pre)
forword = th.function(inputs=[x_seq,bud],outputs=y_seq)

cost = (1+ind-ind)*T.sum((y_seq-y_hat)**2)  #create variable dependency for "ind" variable(theano requirement)
valid = th.function(inputs=[x_seq,y_hat,ind,bud],outputs=cost)
grads = T.grad(cost,parameters,disconnected_inputs='ignore')
            
rnn_train = th.function(inputs=[x_seq,y_hat,ind,bud],outputs=cost,
                       updates=Update_NAG(parameters,grads,ind,Auxiliary,Temp)) #Optimized by NAG

def Validation():
    """Calculate the average edit distance on validation set"""
    Valid_X = []
    for i in range(3000,len(Train_Num)):
        xval = ''; last = ''
        for l in forword(TrainX[Train_Num[Order[i]]],1.25):
            phone = Map[mapping[Map_label(np.argmax(l))]]
            if last != phone:
                xval = xval + phone + ' '
            last = phone
        Valid_X.append(xval[:-1])
        
    Valid_Yhat = Valid_Y()
    return np.mean([edit(Valid_X[i] ,Valid_Yhat[i]) for i in range(len(Valid_X))])

def Valid_Cost():
    """Calculate the cost of the validation set"""
    N = 0;C = 0
    for i in range(3000,len(Train_Num)):
        C += valid(TrainX[Train_Num[Order[i]]],TrainY[Train_Num[Order[i]]],1,1.25)
        N += len(TrainX[Train_Num[Order[i]]])
    return C/N

def Permutate(n):
    """Auxiliary function for making batch of each epoch"""
    s = np.random.permutation(n)
    for i in range(n):
        yield s[i]

print("Model constructed, %f seconds. Start training..."%(time.clock()-st))

Record=[]; Cost = []; Valid_C = []
for j in range(5):
    C = 0; N = 0; batch = 0
    V = Permutate(3000)
    for i in range(3000):
        index = next(V)
        C += rnn_train(TrainX[Train_Num[Order[index]]],TrainY[Train_Num[Order[index]]],batch,1)
        N += len(TrainX[Train_Num[Order[index]]])
        Cost.append(C/N)
        if i%500 == 499:
            Valid_C.append(Valid_Cost())
            print('Cost on last 500 data: %f ; Validation Cost: %f'%(C/N,Valid_C[-1]))
            C = 0; N = 0
            if j > 2:
                val = Validation()
                Record.append(val);
                print("Edit distance on Validation Set: %f" %(val))
                #if len(Record)>1 and Record[-2]+0.5 <= Record[-1]:
                #    break

        batch = batch+1 if batch != (b-1) else 0
    print('%d Epoch(s) trained, %f seconds passed from start.'%((j+1),time.clock()-st))

print("Done training. Using %f seconds." % (time.clock()-st))

print("Start predicting...")

Test = np.load('prob.out.npz')  #DNN prediction of testing set
Test = Test[Test.files[0]]
TID = np.load('TestID.npz')
TestID = TID[TID.files[0]][0]

TestD , _ , Test_Num = make_vector_sequence(Test,TestID,False)
Prediction = []
for i in range(len(Test_Num)):
    pre = []
    for l in forword(TestD[Test_Num[i]],1.25):
        pre.append(mapping[Map_label(np.argmax(l))])
    Prediction.append(pre)

def check(L,i):
    """Sanity Check function to correct unreasonable predictions"""
    if L[i-1] == L[i+1] and L[i] != L[i-1]: #front == behind != me
        return True
    elif L[i]!= L[i+1] and L[i]!= L[i-1]:  # me, front, behind are different 
        return True
    else:
        return False

for inde, pred in enumerate(Prediction):
    for i in range(1,len(pred)-1):
        if check(pred,i):
            pred[i] = pred[i-1]
    Prediction[j] = pred

import csv
csvfile = open('Vprediction.csv','w',newline='')
write = csv.writer(csvfile, delimiter=',')
write.writerow(['id','phone_sequence'])

for ind, pre in enumerate(Prediction):
    predict = ''; last = ''
    for phone in pre:
        if last != phone:
            predict = predict + Map[phone]
        last = phone
    write.writerow([Test_Num[ind],predict[1:-1]])
    
print("All Done.")
