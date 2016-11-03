#Bi-directional Long Short Term Memory (Momentum, NAG, RMSProp, Dropout)
import theano as th
import theano.tensor as T
import numpy as np
import time
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams
from collections import defaultdict

start_time = time.clock()
Data = np.load('prob_t.npz') #Output of DNN
Data = Data[Data.files[0]]
F = np.load('TrainID.npz')
TrainID = F[F.files[0]][0]

Label = {}
Y = open('train.lab')
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

#LSTM structure
st = time.clock()
bat = 1 #Batch size
learing_rate = 0.00003
srng = RandomStreams(seed=5432)

x_seq = T.fmatrix()
y_hat = T.fmatrix()
ind = T.scalar() #Help to do minibatch
bud = T.scalar() #Help to do Dropout 

##### Initialize parameters #####
cons = 0.001; a=0; s=0.01; neuron = 48
a_0 = th.shared(0*np.random.randn(neuron))
h_0 = th.shared(0*np.random.randn(neuron))
Wi = th.shared(s*np.random.randn(48,neuron)-a)
Wf = th.shared(s*np.random.randn(48,neuron)+0.005-a)
Wo = th.shared(s*np.random.randn(48,neuron)-a)
W = th.shared(s*np.random.randn(48,neuron)-a)
Ui = th.shared(s*np.random.randn(neuron,neuron)-a)
Uf = th.shared(s*np.identity(neuron)-a)
Uo = th.shared(s*np.random.randn(neuron,neuron)-a)
U = th.shared(s*np.random.randn(neuron,neuron)-a)
Vi = th.shared(s*np.random.randn(neuron,neuron)-a)
Vf = th.shared(s*np.identity(neuron)-a)
Vo = th.shared(s*np.random.randn(neuron,neuron)-a)
bi = th.shared(cons*np.random.randn(neuron))#-a)
bf = th.shared(cons*np.random.randn(neuron))#-a)
bo = th.shared(cons*np.random.randn(neuron))#-a)
b = th.shared(cons*np.random.randn(neuron))#-a)
#Wout = th.shared(cons*np.random.randn(2*neuron,48)-a)

Wi1f = th.shared(s*np.random.randn(2*neuron,neuron)-a)
Wf1f = th.shared(s*np.random.randn(2*neuron,neuron)+0.005-a)
Wo1f = th.shared(s*np.random.randn(2*neuron,neuron)-a)
W1f = th.shared(s*np.random.randn(2*neuron,neuron)-a)
Wi1b = th.shared(s*np.random.randn(2*neuron,neuron)-a)
Wf1b = th.shared(s*np.random.randn(2*neuron,neuron)+0.005-a)
Wo1b = th.shared(s*np.random.randn(2*neuron,neuron)-a)
W1b = th.shared(s*np.random.randn(2*neuron,neuron)-a)
Ui1 = th.shared(s*np.random.randn(neuron,neuron)-a)
Uf1 = th.shared(s*np.identity(neuron)-a)
Uo1 = th.shared(s*np.random.randn(neuron,neuron)-a)
U1 = th.shared(s*np.random.randn(neuron,neuron)-a)
Vi1 = th.shared(s*np.random.randn(neuron,neuron)-a)
Vf1 = th.shared(s*np.identity(neuron)-a)
Vo1 = th.shared(s*np.random.randn(neuron,neuron)-a)
bi1f = th.shared(cons*np.random.randn(neuron))#-a)
bf1f = th.shared(cons*np.random.randn(neuron))#-a)
bo1f = th.shared(cons*np.random.randn(neuron))#-a)
b1f = th.shared(cons*np.random.randn(neuron))#-a)
bi1b = th.shared(cons*np.random.randn(neuron))#-a)
bf1b = th.shared(cons*np.random.randn(neuron))#-a)
bo1b = th.shared(cons*np.random.randn(neuron))#-a)
b1b = th.shared(cons*np.random.randn(neuron))#-a)
W1out = th.shared(cons*np.random.randn(2*neuron,48)-a)

Auxiliary = []; Temp = []
parameters = [Wi,Wf,Wo,W,bi,bf,bo,b,U,Ui,Uf,Uo,Vi,Vf,Vo,Wi1f,Wf1f,Wo1f,W1f,Wi1b,Wf1b,Wo1b,W1b,bi1f,bf1f,bo1f,b1f
              ,bi1b,bf1b,bo1b,b1b,U1,Ui1,Uf1,Uo1,Vi1,Vf1,Vo1,W1out]
for param in parameters:
    Auxiliary.append(th.shared(np.zeros(param.get_value().shape)))
    Temp.append(th.shared(np.zeros(param.get_value().shape)))

##### Do gradient clip #####
c = 1
Wi = th.gradient.grad_clip(Wi,-c,c)
bi = th.gradient.grad_clip(bi,-c,c)
Wf = th.gradient.grad_clip(Wf,-c,c)
bf = th.gradient.grad_clip(bf,-c,c)
Wo = th.gradient.grad_clip(Wo,-c,c)
bo = th.gradient.grad_clip(bo,-c,c)
W = th.gradient.grad_clip(W,-c,c)
b = th.gradient.grad_clip(b,-c,c)
Ui = th.gradient.grad_clip(Ui,-c,c)
Uf = th.gradient.grad_clip(Uf,-c,c)
Uo = th.gradient.grad_clip(Uo,-c,c)
U = th.gradient.grad_clip(U,-c,c)
Vi = th.gradient.grad_clip(Vi,-c,c)
Vf = th.gradient.grad_clip(Vf,-c,c)
Vo = th.gradient.grad_clip(Vo,-c,c)
#Wout = th.gradient.grad_clip(Wout,-c,c)

Wi1f = th.gradient.grad_clip(Wi1f,-c,c)
bi1f = th.gradient.grad_clip(bi1f,-c,c)
Wf1f = th.gradient.grad_clip(Wf1f,-c,c)
bf1f = th.gradient.grad_clip(bf1f,-c,c)
Wo1f = th.gradient.grad_clip(Wo1f,-c,c)
bo1f = th.gradient.grad_clip(bo1f,-c,c)
W1f = th.gradient.grad_clip(W1f,-c,c)
b1f = th.gradient.grad_clip(b1f,-c,c)
Wi1b = th.gradient.grad_clip(Wi1b,-c,c)
bi1b = th.gradient.grad_clip(bi1b,-c,c)
Wf1b = th.gradient.grad_clip(Wf1b,-c,c)
bf1b = th.gradient.grad_clip(bf1b,-c,c)
Wo1b = th.gradient.grad_clip(Wo1b,-c,c)
bo1b = th.gradient.grad_clip(bo1b,-c,c)
W1b = th.gradient.grad_clip(W1b,-c,c)
b1b = th.gradient.grad_clip(b1b,-c,c)
Ui1 = th.gradient.grad_clip(Ui1,-c,c)
Uf1 = th.gradient.grad_clip(Uf1,-c,c)
Uo1 = th.gradient.grad_clip(Uo1,-c,c)
U1 = th.gradient.grad_clip(U1,-c,c)
Vi1 = th.gradient.grad_clip(Vi1,-c,c)
Vf1 = th.gradient.grad_clip(Vf1,-c,c)
Vo1 = th.gradient.grad_clip(Vo1,-c,c)
W1out = th.gradient.grad_clip(W1out,-c,c)

##### Optimization methods #####
def Update_Momentum(para,grad,ind,Momentum,Temp):
    """theano update, optimized by Momentum"""
    updates = []; off_on = ifelse(T.lt(ind,bat-1),0,1)
    for ix in range(len(grad)):
        gradient = T.clip(grad[ix],-1,1)#T.clip(grad[ix],np.float32(-0.01),np.float32(0.01))
        direction = (0.95)*Momentum[ix] - (learing_rate/bat)*(gradient+Temp[ix])
        updates.append((para[ix], para[ix]+direction*off_on))
        updates.append((Momentum[ix], Momentum[ix]*(1-off_on)+direction*off_on))
        updates.append((Temp[ix], (Temp[ix]+gradient)*(1-off_on)))
    return updates

def Update_NAG(para,grad,ind,Real,Temp):
    """theano update, optimized by NAG"""
    updates = []; off_on = ifelse(T.lt(ind,bat-1),0,1)
    for ix in range(len(grad)):
        grad[ix] = T.clip(grad[ix],-1,1)
        gradient = -(learing_rate/bat)*(grad[ix]+Temp[ix])
        spy_position = (1+0.95)*(para[ix]+gradient)-0.95*Real[ix]
        updates.append((para[ix], (spy_position)*off_on+para[ix]*(1-off_on)))
        updates.append((Real[ix], (para[ix]+gradient)*off_on+Real[ix]*(1-off_on)))
        updates.append((Temp[ix], (Temp[ix]+grad[ix])*(1-off_on)))
    return updates

def Update_RMSProp(para,grad,ind,Sigma_square,Temp):
    """theano update, optimized by RMSProp"""
    updates = []; off_on = ifelse(T.lt(ind,bat-1),0,1); alpha = 0.9
    for ix in range(len(grad)):
        #grad[ix] = T.clip(grad[ix],-1,1)
        gradient = (grad[ix]+Temp[ix])/bat
        Factor = Sigma_square[ix]*alpha+(1-alpha)*gradient**2
        direction = -(learing_rate)*gradient/T.sqrt(Factor)
        updates.append((para[ix], (para[ix]+direction)*off_on+para[ix]*(1-off_on)))
        updates.append((Sigma_square[ix], Factor*off_on+Sigma_square[ix]*(1-off_on)))
        updates.append((Temp[ix], (Temp[ix]+grad[ix])*(1-off_on)))
    return updates

##### Activation Functions #####
def sigmoid(Z):
    return 1/(1+T.exp(-Z))

def ReLU(Z):
    return T.switch(Z<0,0,Z)

def tanh(Z):
    return T.tanh(Z)

def softmax(Z):
    z = T.exp(Z)
    return (z.T/T.sum(z,axis=1)).T

def step(z_t,zi_t,zf_t,zo_t,c_tm1,h_tm1):
    Z_t = tanh(z_t + T.dot(h_tm1,U))
    Zi_t = sigmoid(zi_t + T.dot(h_tm1,Ui) + T.dot(c_tm1,Vi))
    Zf_t = sigmoid(zf_t + T.dot(h_tm1,Uf) + T.dot(c_tm1,Vf)) 
    c_t = Z_t*Zi_t + c_tm1*Zf_t
    Zo_t = sigmoid(zo_t + T.dot(h_tm1,Uo) + T.dot(c_t,Vo))
    h_t = tanh(c_t)*Zo_t
    return c_t, h_t

def step2(z_t,zi_t,zf_t,zo_t,c_tm1,h_tm1):
    Z_t = tanh(z_t + T.dot(h_tm1,U1))
    Zi_t = sigmoid(zi_t + T.dot(h_tm1,Ui1) + T.dot(c_tm1,Vi1))
    Zf_t = sigmoid(zf_t + T.dot(h_tm1,Uf1) + T.dot(c_tm1,Vf1)) 
    c_t = Z_t*Zi_t + c_tm1*Zf_t
    Zo_t = sigmoid(zo_t + T.dot(h_tm1,Uo1) + T.dot(c_t,Vo1))
    h_t = tanh(c_t)*Zo_t
    return c_t, h_t

##### Structure Building #####
z_seq = T.dot(x_seq,W)+b.dimshuffle('x',0)
zi_seq = T.dot(x_seq,Wi)+bi.dimshuffle('x',0)
zo_seq = T.dot(x_seq,Wo)+bo.dimshuffle('x',0)
zf_seq = T.dot(x_seq,Wf)+bf.dimshuffle('x',0)

# Forward direction
[cf_seq,hf_seq],_ = th.scan(step, sequences = [z_seq,zi_seq,zf_seq,zo_seq], 
                               outputs_info = [a_0,h_0],
                              truncate_gradient=-1)
# Backward direction
[cb_seq,hb_seq],_ = th.scan(step, sequences = [z_seq[::-1],zi_seq[::-1],zf_seq[::-1],zo_seq[::-1]], 
                               outputs_info = [a_0,h_0],
                              truncate_gradient=-1)

y1_seq = T.concatenate([hf_seq,hb_seq[::-1]],axis=1)
#h_seq = ifelse(T.lt(bud,1.05), h_out*srng.binomial(size=T.shape(h_out),p=0.8),h_out)/bud  #Dropout
#y1_seq = softmax(T.dot(h_out,Wout))*(bud/bud)

z1f_seq = T.dot(y1_seq,W1f)+b1f.dimshuffle('x',0)
zi1f_seq = T.dot(y1_seq,Wi1f)+bi1f.dimshuffle('x',0)
zo1f_seq = T.dot(y1_seq,Wo1f)+bo1f.dimshuffle('x',0)
zf1f_seq = T.dot(y1_seq,Wf1f)+bf1f.dimshuffle('x',0)

z1b_seq = T.dot(y1_seq,W1b)+b1b.dimshuffle('x',0)
zi1b_seq = T.dot(y1_seq,Wi1b)+bi1b.dimshuffle('x',0)
zo1b_seq = T.dot(y1_seq,Wo1b)+bo1b.dimshuffle('x',0)
zf1b_seq = T.dot(y1_seq,Wf1b)+bf1b.dimshuffle('x',0)

# Forward direction
[cf1_seq,hf1_seq],_ = th.scan(step2, sequences = [z1f_seq,zi1f_seq,zf1f_seq,zo1f_seq], 
                               outputs_info = [a_0,h_0],
                              truncate_gradient=-1)
# Backward direction
[cb1_seq,hb1_seq],_ = th.scan(step2, sequences = [z1b_seq[::-1],zi1b_seq[::-1],zf1b_seq[::-1],zo1b_seq[::-1]], 
                               outputs_info = [a_0,h_0],
                              truncate_gradient=-1)

h1_out = T.concatenate([hf1_seq,hb1_seq[::-1]],axis=1)
y_seq = softmax(T.dot(h1_out,W1out))+bud*0
forword = th.function(inputs=[x_seq,bud],outputs=y_seq)

cost = (1+ind-ind)*T.sum((y_seq-y_hat)**2)
valid = th.function(inputs=[x_seq,y_hat,ind,bud],outputs=cost)
grads = T.grad(cost,parameters)#,disconnected_inputs='ignore')
    
rnn_train = th.function(inputs=[x_seq,y_hat,ind,bud],outputs=cost,
                       updates=Update_NAG(parameters,grads,ind,Auxiliary,Temp))

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
    return np.mean([edit(Valid_X[i] ,Valid_Yhat[i]) for i in range(len(Valid_X))]).astype(np.float32)

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

        batch = batch+1 if batch != (bat-1) else 0
    print('%d Epoch(s) trained, %f seconds passed from start.'%((j+1),time.clock()-st))

end = time.clock()
print("Done training. Using %f seconds." % (end-st))

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
    Prediction[inde] = pred

import csv
csvfile = open('LSTM_predict.csv','w',newline='')
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