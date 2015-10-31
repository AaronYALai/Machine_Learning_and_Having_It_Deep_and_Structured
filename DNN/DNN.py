import theano as th
import theano.tensor as T
import numpy as np
import time
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams

st = time.clock()
O = open('train.ark')
Raw = O.readlines()
O.close()

def memo(f): 
    """Memoization decorator, Used to accelerate the retrieval"""
    cache = {}
    def _f(*args):
        try:
            return cache[args]
        except KeyError:
            cache[args] = result = f(*args)
            return result
        except TypeError: #Some elements of args unhashable
            return f(args)
    _f.cache = cache
    return _f

def trans(Raw,Tid,Data):
    """Transform data into an dictionary of ID and a list of features""" 
    for i in range(len(Raw)):
        line = Raw[i].strip().split()
        Tid[i] = line.pop(0)
        Data.append(list(map(float,line)))
    
def Normalize(Data):
    """Standardize the data"""
    n = len(Data)
    D_T = np.transpose(Data)
    Mean, Std = zip(*[(np.mean(D_T[i]),np.std(D_T[i])) for i in range(len(D_T))])
    Mean, Std = np.array(Mean),np.array(Std)
    for i in range(n):
        Data[i] = (Data[i] - Mean)/Std


TrainID = {}
TrainData = []
trans(Raw, TrainID, TrainData)
TrainData = np.array(TrainData).astype(dtype='float32')
Normalize(TrainData)

#Load the label data as a dictionary
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

print("Data loaded, using %f seconds"%(time.clock()-st))

def Update(para,grad):
    """theano update auxiliary function"""
    return [(para[ix], para[ix]+Direction(ix,grad[ix])) for ix in range(len(grad))]

Momentum = {}
def Direction(ix,grad):
    """Compute the update"""
    try:
        Momentum[ix] = update = 0.98*Momentum[ix] - 0.01*grad
        return update
    except:
        Momentum[ix] = update = -0.01*grad
        return update


N = 1100131              #Training set size
batchsize = 40            
cons = np.float32(0.033)   #Scaling for random normal initializing 
archi = 1280              #Architecture - number of neurons
srng = RandomStreams(seed=5432)

x = T.fmatrix()
y_hat = T.fmatrix()
bud = T.scalar() #Switcher to decide dropout of neurons or not
W1 = th.shared(cons*np.random.randn(621,archi).astype(np.float32))
b1 = th.shared(cons*np.random.randn(archi).astype(np.float32))

W2 = th.shared(cons*np.random.randn(archi/2,archi).astype(np.float32))
b2 = th.shared(cons*np.random.randn(archi).astype(np.float32))

W3 = th.shared(cons*np.random.randn(archi/2,archi).astype(np.float32))
b3 = th.shared(cons*np.random.randn(archi).astype(np.float32))

W4 = th.shared(cons*np.random.randn(archi/2,archi).astype(np.float32))
b4 = th.shared(cons*np.random.randn(archi).astype(np.float32))

W5 = th.shared(cons*np.random.randn(archi/2,48).astype(np.float32))
b5 = th.shared(cons*np.random.randn(48).astype(np.float32))


def Maxout(Z,bud):
    Z_out = T.maximum(Z[:,:int(archi/2)],Z[:,int(archi/2):])
    return ifelse(T.lt(bud,1.05), Z_out*srng.binomial(size=T.shape(Z_out),p=0.8).astype('float32'),Z_out)   

def Softmax(z):
    Z = T.exp(z)
    results,_ = th.scan(lambda x: x/T.sum(x),sequences=Z)
    return results

z1 = T.dot(x,W1) + b1.dimshuffle('x',0)
a1 = Maxout(z1,bud)/bud         #Scaling of output while predicting 

z2 = T.dot(a1,W2) + b2.dimshuffle('x',0)
a2 = Maxout(z2,bud)/bud

z3 = T.dot(a2,W3) + b3.dimshuffle('x',0)
a3 = Maxout(z3,bud)/bud

z4 = T.dot(a3,W4) + b4.dimshuffle('x',0)
a4 = Maxout(z4,bud)/bud

z5 = T.dot(a4,W5) + b5.dimshuffle('x',0)

y = Softmax(z5/bud)

forward = th.function([x,bud],y) #Bud to indicate no dropout
parameters = [W1,b1,W2,b2,W3,b3,W4,b4,W5,b5] 
Cost = ((-T.log((y*y_hat).sum(axis=1))).sum())/batchsize

grads = T.grad(Cost,parameters,disconnected_inputs='ignore')
gradient_2 = th.function(inputs=[x,y_hat,bud],updates=Update(parameters,grads),outputs=Cost)

@memo
def y_hat(i):
    """give the np array of y_hat"""
    l = np.zeros(48,dtype=np.float32)
    l[lab_index[Label[TrainID[i]]]] = 1
    return l

def Permutate(n):
    """Auxiliary function for making batch of each epoch"""
    s = np.random.permutation(n)
    for i in range(n):
        yield s[i] 

def gen_data(ind,Data,Tid,low,up):
    if ind > 3+low:
        if ind < up-4:
            for j in range(4):
                if Tid[ind+1+j][:10]!= Tid[ind][:10]:
                    return np.append(Data[ind-4 : ind+1+j],np.zeros(69*(4-j)))
                if Tid[ind-1-j][:10]!= Tid[ind][:10]:
                    return np.append(np.zeros(69*(4-j)),Data[ind-j : ind+5])
            return np.ravel(Data[ind-4 : ind+5])
        else:
            return np.append(Data[ind-4 : up],np.zeros(69*(ind+5-up)))
    else:
        return np.append(np.zeros(69*(4+low-ind)),Data[low: ind+5])

def Accuracy(low,up):
    inp = [gen_data(i,TrainData,TrainID,low,up) for i in range(low,up)]
    Yy = forward(inp ,np.float32(1.25))
    return sum([1 for i in range(low,up) if np.argmax(Yy[i-low]) == lab_index[Label[TrainID[i]]]])/(up-low)

print("Start Training...")

Accuracy_Record = []
Cost_Record = []   
for j in range(20):
    V = Permutate(N)
    Costs = 0
    for i in range(int(N/batchsize)):
        batch_X = []; batch_Y = []
        for k in range(batchsize):
            index = next(V)    #retrieve data in a random order 
            batch_X.append(gen_data(index,TrainData,TrainID,0,N))
            batch_Y.append(y_hat(index))
        Costs += gradient_2(batch_X,batch_Y,1)

    Cost_Record.append(Costs/int(N/batchsize))
    Accuracy_Record.append(Accuracy(1100131,1124823)) #Validation set 
    print("Cost: %f" % Cost_Record[-1],", %f seconds used."%(time.clock()-st))
    if (Accuracy_Record[0]!= Accuracy_Record[-1]) and (Accuracy_Record[-2] - 0.005 > Accuracy_Record[-1]): 
        print("Validation accuracy starts decreasing, stop training")
        break     

print("Training Accuracy: %f" % Accuracy(0,N))
print("Validation Accuracy: %f" % Accuracy(1100131,1124823))
end = time.clock()
print("Done, Using %f seconds." % (end-st))

print("Start predicting...")

O = open('test.ark')
TEST = O.readlines()
O.close()

TestID = {}
TestData = []
trans(TEST, TestID, TestData)

TestData = np.array(TestData).astype(np.float32)
Normalize(TestData)
Test_input = [gen_data(i,TestData,TestID,0,len(TestData)) for i in range(len(TestData))]
Y_test = forward(Test_input,np.float32(1.25))
np.savez('prob.out',Y_test)

Mapp = open('48_39.map')
mapp = Mapp.readlines()
Mapp.close()

def Map_label(n):
    for key,val in lab_index.items():
        if val == n:
            return key
    print("Worng!")

mapping = {}
for line in mapp:
    line = line.strip().split('\t')
    mapping[line[0]] = line[1]

Predict = {}
for i in range(len(TestData)):
    Predict[i] = mapping[Map_label(np.argmax(Y_test[i]))]

#Writing out predictions
import csv
with open('Predicts.csv','w',newline='') as csvfile:
    write = csv.writer(csvfile, delimiter=',')
    write.writerow(['Id','Prediction'])
    for i in range(len(Predict)):
        write.writerow([TestID[i],Predict[i]])

with open('Records.csv','w',newline='') as csvfile2:
    write2 = csv.writer(csvfile2, delimiter=',')
    write2.writerow(Cost_Record)
    write2.writerow(Accuracy_Record)

print("All Done. %f seconds."%(time.clock()-st)) 
