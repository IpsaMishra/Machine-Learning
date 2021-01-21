import numpy as np
import matplotlib.pyplot as plt

def data_subset(raw_data_list, class_list):
    subset_data_list = []
    for i in range(len(raw_data_list)):
        for j in range(len(class_list)):
            if raw_data_list[i][1] == class_list[j]:
                each_data = []
                each_data.append(raw_data_list[i][0])
                each_data.append(j)
                subset_data_list.append(each_data)
            else:
                continue
    print("subset_data_list len: ", len(subset_data_list))
    return subset_data_list

def generate_data(mu, sigma, n_samples):
    x = np.random.multivariate_normal(mu, sigma, n_samples)
    return x

def pred_calc(w,X):
    z = np.matmul(X,w)
    y = 1 / (1 + np.exp(-z))
    return y

def loss_calc(y,t):
    l = np.sum(- t * np.log(y) - (1 - t) * np.log(1 - y))
    return l / np.size(t)

def grad_calc(X,y,t):
    d = y - t
    if np.size(t) == 1:
        g = d * X
    else:
        g = np.matmul(np.transpose(X),d)
    return g / np.size(t)


def roc_auc(y, pred):
    # pred is the prediction probabilities and y is the true label set
    y = np.array(y)
    pred = np.array(pred)
    tpr = np.zeros(101)
    fpr = np.zeros(101)
    tpr[0] = 1
    fpr[0] = 1
    tot_pos = np.sum(y)
    tot_neg = np.sum(1 - y)
    for i in range(1,100):
        thr = i / 100
        p = (pred > thr) * 1
        tp = np.sum(2 * y + p == 3)
        fp = np.sum(2 * y + p == 1)
        tpr[i] = tp / tot_pos
        fpr[i] = fp / tot_neg
    
    auc = 0     # Initialize auc for roc curve
    for i in range(100):
        auc += 0.5 * (tpr[i] + tpr[i + 1]) * (fpr[i] - fpr[i + 1])

    return tpr, fpr, auc
    

def LogReg(Xtrain,ytrain):
    loss = []
    grad = []
    n_weight = np.shape(Xtrain)[1]
    w = np.random.random(n_weight)
    y = pred_calc(w,Xtrain)
    l = loss_calc(y,ytrain)
    loss.append(l)
    g = grad_calc(Xtrain,y,ytrain)
    grad.append(np.mean(np.abs(g)))
    print(lr)
    for i in range(maxiter):
        w = w - lr * g
        y = pred_calc(w,Xtrain)
        le = loss_calc(y,ytrain)
        loss.append(le)
        g = grad_calc(Xtrain,y,ytrain)
        grad.append(np.mean(np.abs(g)))
        if np.abs(le - l) <= lr * L1diff or np.mean(np.abs((g))) <=0.01:
            break
        l = le
    print('Number of epochs:',i+1)
    
    loss = np.array(loss)
    grad = np.array(grad)
    #plt.figure()
    #plt.plot(loss)
    #plt.figure()
    #plt.plot(grad)
    return w, loss, grad

lr = 1
maxiter = 10000
L1diff = 0.0001
mu1 = np.array([1,0])
mu2 = np.array([0,1])
sigma1 = np.array([[1,0.75],[0.75,1]])
sigma2 = np.array([[1,-0.5],[0.5,1]])
n_train = 500
n_test = 500

Xtrain1 = generate_data(mu1,sigma1,n_train)
Xtrain2 = generate_data(mu2,sigma2,n_train)
Xtrain = np.concatenate((Xtrain1,Xtrain2))
del Xtrain1, Xtrain2
ytrain = np.concatenate((np.ones(n_train),np.zeros(n_train)))
Xtrain = np.append(Xtrain,np.ones((np.size(ytrain),1)),axis=1)
    
Xtest1 = generate_data(mu1,sigma1,n_test)
Xtest2 = generate_data(mu2,sigma2,n_test)
Xtest = np.concatenate((Xtest1,Xtest2))
del Xtest1, Xtest2
ytest = np.concatenate((np.ones(n_test),np.zeros(n_test)))
Xtest = np.append(Xtest,np.ones((np.size(ytest),1)),axis=1)
    
w, loss, grad = LogReg(Xtrain,ytrain)
y = 0.0 + (pred_calc(w,Xtest) > 0.5)
    
#plt.figure()
#plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
#x1 = -3 + 6 * np.arange(10000) / 10000
#x2 = -(w[0] * x1 + w[2]) / w[1]
#plt.plot(x1,x2)

pred = pred_calc(w,Xtest)
y = 0.0 + (pred > 0.5)
print('Accuracy: ', np.sum(1-np.abs(y-ytest))/len(Xtest))

tpr, fpr, auc = roc_auc(ytest, pred)
print('AUC-ROC value is ', auc)
plt.figure()
plt.plot(fpr,tpr)

l_rate = [0.0001,0.001,0.01,0.1,1]
iter = [10000,10000,2641,273,29]
plt.figure()
plt.plot(l_rate,iter)