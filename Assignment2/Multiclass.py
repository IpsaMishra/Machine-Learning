from mxnet import gluon
import numpy as np
import matplotlib.pyplot as plt

def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

def pred_calc(w,X):
    z = np.matmul(X,w)
    exp = np.exp(z - np.max(z, axis = 1).reshape((-1,1)))
    norm = np.sum(exp, axis=1).reshape((-1,1))
    h = exp / norm
    return h

def pred_class(w,X):
    h = pred_calc(w,X)
    yhat = []
    n = np.shape(h)[0]
    for i in range(n):
        yhat.append(np.argmax(h[i]))
    return yhat

def loss_calc(y,t):
    n = len(t)
    loss = 0
    for i in range(n):
        loss += - np.log(y[i,np.int(t[i])])
    return loss/n

def grad_calc(X,y,t):
    # X should have a shape m x dim
    # y should have a shape m x nClass
    # t should have values from 0 through nClass - 1 and be of len m
    
    X = np.array(X)
    y = np.array(y)
    m = np.shape(y)[0]
    nClass = np.shape(y)[1]
    
    nt = np.zeros((m,nClass))
    for i in range(m):
        nt[i, np.int(t[i])] = 1  
    diff = y - nt
    grad = np.matmul(np.transpose(X), diff) / m
    return grad

def MultiLogReg(Xtrain, ytrain, maxiter, lr, L1diff, gradthres):
    loss = []
    grad = []
    dims = np.shape(Xtrain)[1]
    nclass = len(np.unique(ytrain))
    w = np.random.random((dims,nclass))
    y = pred_calc(w,Xtrain)
    l = loss_calc(y,ytrain)
    g = grad_calc(Xtrain,y,ytrain)
    loss.append(l)
    grad.append(np.mean(np.abs(g)))
    for i in range(maxiter):
        w = w - lr * g
        y = pred_calc(w, Xtrain)
        le = loss_calc(y, ytrain)
        loss.append(le)
        g = grad_calc(Xtrain,y,ytrain)
        grad.append(np.mean(np.abs(g)))
        print('Epoch ==> ', i, 'Loss = ', le, ', Absolute Gradient Sum = ', np.mean(np.abs(g)))
        if np.abs(le - l) <= lr * L1diff or np.mean(np.abs(g)) < gradthres:
            break
        l = le
    print('Number of epochs: ', i+1)
    loss = np.array(loss)
    grad = np.array(grad)
    return w, loss, grad

def multiclassMetrics(y, yhat):
    # y is the true label data, Example y = 1
    # yhat is the predicted class label. Example yhat = 4
    # Assumes that the classes are indexed from 0 to nClass - 1
    
    y = np.array(y)
    yhat = np.array(yhat)
    
    nClass = len(np.unique(y))  # Total number of classes
    conf_matr = np.zeros((nClass, nClass))
    PrecisionVal = np.zeros(nClass)
    RecallVal = np.zeros(nClass)
    
    if len(y) != len(yhat):
        print('True label set and Predicted class label set size must match')
        return None
    
    Match = 0
    m = len(y)
    for i in range(m):
        conf_matr[y[i], yhat[i]] += 1
        if y[i] == yhat[i]:
            Match += 1
    
    for i in range(nClass):
        PrecisionVal[i] = conf_matr[i,i] / np.sum(conf_matr[:,i])
        RecallVal[i] = conf_matr[i,i] / np.sum(conf_matr[i,:])
    
    accr = Match / np.sum(conf_matr)
    return PrecisionVal, RecallVal, accr


# Initialize parameters
lr = 0.1
maxiter = 1000
L1diff = 0.001
gradthres = 0.0005

# Process dataset
mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)

Xtrain = []
ytrain = []
for data in mnist_train:
    x, y = data
    if y <= 4:      # Filter out data with class label 5 or more
        x = x.reshape(1,784).asnumpy()
        x = x[0]
        Xtrain.append(x)
        ytrain.append(y)
Xtrain = np.array(Xtrain)
Xtrain = np.concatenate((Xtrain,np.ones((np.shape(Xtrain)[0],1))),axis=1)
ytrain = [np.int(val) for val in ytrain]

Xtest = []
ytest = []
for data in mnist_test:
    x, y = data
    if y <= 4:      # Filter out data with class label 5 or more
        x = x.reshape(1,784).asnumpy()
        x = x[0]
        Xtest.append(x)
        ytest.append(y)    
Xtest = np.array(Xtest)
Xtest = np.concatenate((Xtest,np.ones((np.shape(Xtest)[0],1))),axis=1)
ytest = [np.int(val) for val in ytest]

# Train Multi-class logistic model
w, loss, grad = MultiLogReg(Xtrain, ytrain, maxiter, lr, L1diff, gradthres)

# Plot loss and L1 norm of gradient matrix
plt.figure()
plt.plot(loss)

plt.figure()
plt.plot(grad)

# Evaluate the trained model
yhat = pred_class(w,Xtest)
p,r,a = multiclassMetrics(ytest, yhat)
print('Accuracy = ', a)
for i in range(5):
    print('Precision for class ', i, 'is ', p[i])
for i in range(5):
    print('Recall for class ', i, 'is ', r[i])