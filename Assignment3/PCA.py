import numpy as np
from keras.datasets import mnist
import time
import matplotlib.pyplot as plt

## Q2.1
def myPCA(X ,k):
    M = np.mean(X , axis = 0)               # Calculate Mean
    N = X - M                               # Normalize data
    C = np.cov(N , rowvar = False)          # Calculate covariance (trick)
    eig_vals, eig_vecs = np.linalg.eigh(C)  # Eigen computation for symmetric matrix
    ind = np.argsort(eig_vals)[::-1]        # Descending order for eigen-values
    eig_vecs = eig_vecs[ind, :]             # Sort eigen-vectors
    top_eig = eig_vecs[0:k, :]              # Choose top k eigen-vectors
    X_kPC = np.matmul(N, np.transpose(top_eig))       # Projected data
    return X_kPC, top_eig, M

## Test Example
X = [[1,2,3],[1,4,9],[1,8,27],[1,16,81]]
k = 2
X_kPC, top_eig, M = myPCA(X, k)
X_recons = np.matmul(X_kPC, top_eig) + M

## Q2.2
k = 2

# Split the mnist data into train and test
(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()

# Reshape the dataset
train_size = 60000              #Number of training images to train the model
test_size = 10000               #Number of testing images to test the model
v_length = 784                  #Dimension of flattened input image size i.e. if input image size is [28x28], then v_length = 784
Xtrain = Xtrain.reshape(train_size, v_length)       #Reshape the array as we need it without changing the data
Xtest = Xtest.reshape(test_size, v_length)          #Reshape the array as we need it without changing the data
Xtrain = Xtrain.astype("float32")                   #Converting to floating point representations
Xtest = Xtest.astype("float32")                     #Converting to floating point representations
Xtrain /= 255                                       #Normalize the floating point variables in the range (0,1)
Xtest /= 255                                        #Normalize the floating point variables in the range (0,1)
start_time = time.time()
Xtrain_kPC, top_eig, M = myPCA(Xtrain, k)
end_time = time.time()
print('Time taken to perform PCA on MNIST train data with', k, 'dimensions is ', end_time - start_time, 'seconds')

## Q2.3
if k < 2:
    k = 2
    Xtrain_kPC, top_eig, M = myPCA(Xtrain, k)

x = Xtrain_kPC[:,0]
y = Xtrain_kPC[:,1]
plt.scatter(x, y, alpha = 0.5, c = ytrain)
plt.show()

## Q2.4
k = 10
n_images = 4

Xtrain_kPC, top_eig, M = myPCA(Xtrain, k)
Xtrain_recon = np.matmul(Xtrain_kPC, top_eig) + M
for j in range(n_images):
    print('Image for numeral ',ytrain[j])
    sample = Xtrain[j].reshape(28,28)
    plt.figure()
    plt.imshow(sample, cmap='gray')
    sample_PCA = Xtrain_recon[j].reshape(28,28)
    plt.figure()
    plt.imshow(sample_PCA, cmap='gray')

## Q2.5
lr = 1
maxiter = 10
L1diff = 0.001
gradthres = 0.0005
k = 30     #Number of dimensions in PCA

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

Xtrain_kPC, top_eig, M = myPCA(Xtrain, k)
Xtest_kPC, top_eig2, M2 = myPCA(Xtest, k)

start_time = time.time()
w, loss, grad = MultiLogReg(Xtrain, ytrain, maxiter, lr, L1diff, gradthres)
end_time = time.time()

yhat = pred_class(w,Xtrain)
p,r,a = multiclassMetrics(ytrain, yhat)
print('Accuracy for raw images on train set = ', a)

yhat = pred_class(w,Xtest)
p,r,a = multiclassMetrics(ytest, yhat)
print('Accuracy for raw images on test set = ', a)
print('Training time for raw images = ', end_time - start_time, ' seconds')

start_time2 = time.time()
w2, loss, grad = MultiLogReg(Xtrain_kPC, ytrain, maxiter, lr, L1diff, gradthres)
end_time2 = time.time()

yhat_kPC = pred_class(w2,Xtrain_kPC)
p,r,a = multiclassMetrics(ytrain, yhat_kPC)
print('Accuracy for raw images on train set = ', a)

yhat_kPC = pred_class(w2, Xtest_kPC)
p,r,a = multiclassMetrics(ytest, yhat_kPC)
print('Accuracy for projected images on test set = ', a)
print('Training time for projected images = ', end_time2 - start_time2, ' seconds')

## Q2.6
for j in range(10):
    plt.figure()
    plt.imshow(w[:,j].reshape(28,28), cmap='gray')