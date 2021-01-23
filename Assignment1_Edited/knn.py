import numpy as np

def generate_data_class(mu1, sigma1, mu2, sigma2, n_samples):
    d = len(mu1)
    x1 = np.zeros((n_samples, d + 1))
    x2 = np.ones((n_samples, d + 1))
    x1[:, 0:d] = np.random.multivariate_normal(mu1,sigma1,n_samples)
    x2[:, 0:d] = np.random.multivariate_normal(mu2,sigma2,n_samples)
    x = np.concatenate((x1,x2),axis=0)
    return x

def generate_data_regress(mu, sigma, var, n_samples):
    x = np.zeros((n_samples, 3))
    x[:, 0:2] = np.random.multivariate_normal(mu, sigma, n_samples)
    x[:, 2] = 2 * x[:, 0] + x[:, 1] + np.random.normal(0, 0.5, n_samples)
    return x

def myknnclassify(train, test, k):
    d = np.shape(test)[1] - 1
    n_test = len(test)
    n_train = len(train)
    p_test = np.zeros(n_test)
    for i in range(n_test):
        dist = np.zeros(n_train)
        for j in range(n_train):
            dist[j] = np.linalg.norm(train[j,0:d] - test[i,0:d])
        classify = train[dist.argsort()[:k],d]
        p_test[i] = np.round(np.mean(classify))
    return p_test

def myknnregress(train, test, k):
    d = np.shape(test)[1] - 1
    n_test = len(test)
    n_train = len(train)
    p_test = np.zeros(n_test)
    for i in range(n_test):
        dist = np.zeros(n_train)
        for j in range(n_train):
            dist[j] = np.linalg.norm(train[j,0:d] - test[i,0:d])
        regress = train[dist.argsort()[:k],d]
        p_test[i] = np.average(regress, weights = 1 / dist[dist.argsort()[-k:]])
    return p_test

def myLWR(train, test, k):
    d = np.shape(test)[1] - 1
    n_test = len(test)
    n_train = len(train)
    p_test = np.zeros(n_test)
    y_test = test[:,d]
    X = np.zeros((len(test),k))
    for i in range(n_test):
        dist = np.zeros(n_train)
        for j in range(n_train):
            dist[j] = np.linalg.norm(train[j,0:d] - test[i,0:d])
        X[i,:] = train[dist.argsort()[:k],d]
    weight1 = np.linalg.inv(np.matmul(np.transpose(X),X))
    weight2 = np.matmul(np.transpose(X),y_test)
    weight = np.matmul(weight1,weight2)
    p_test = np.matmul(X,weight)
    return p_test, weight

## Part (a): Run knn classifier for k = 1; 2; 3; 4; 5; 10; 20
mu1 = [1,0]
mu2 = [0,1]
sigma1 = [[1,0.75],[0.75,1]]
sigma2 = [[1,-0.5],[0.5,1]]
n_train = 200
n_test = 50
k_list = [1,2,3,4,5,10,20]

train = generate_data_class(mu1,sigma1,mu2,sigma2,n_train)
test = generate_data_class(mu1,sigma1,mu2,sigma2,n_test)
y_test = test[:,np.shape(test)[1] - 1]

print('\n\n Part (a): Run knn classifier for k = 1; 2; 3; 4; 5; 10; 20')
for k in k_list:
    p_test = myknnclassify(train,test,k)
    acc = np.mean(1 - np.abs(y_test - p_test))
    print('For k = ',str(k),' Accuracy is ',acc)
    
del train, test, y_test, p_test


## Part (b): Run knn regressor for k = 1; 2; 3; 5; 10; 20; 50; 100
mu = [1,0]
sigma = [[1,0.75],[0.75,1]]
var = 0.5
n_train = 300
n_test = 100
k_list = [1, 2, 3, 5, 10, 20, 50, 100]

train = generate_data_regress(mu, sigma, var, n_train)
test = generate_data_regress(mu, sigma, var, n_test)
y_test = test[:,np.shape(test)[1] - 1]

print('\n\n Part (b): Run knn regressor for k = 1; 2; 3; 5; 10; 20; 50; 100')
for k in k_list:
    p_test = myknnregress(train,test,k)
    err = np.linalg.norm(y_test - p_test)
    print('For k = ',str(k),' Error is ',err)
    
del train, test, y_test, p_test


## Part (c): Run locally weighted knn regressor for k = 1; 2; 3; 5; 10; 20; 50; 100
mu = [1,0]
sigma = [[1,0.75],[0.75,1]]
var = 0.5
n_train = 300
n_test = 100
k_list = [1, 2, 3, 5, 10, 20, 50, 100]

train = generate_data_regress(mu, sigma, var, n_train)
test = generate_data_regress(mu, sigma, var, n_test)
y_test = test[:,np.shape(test)[1] - 1]

print('\n\n Part (c): Run locally weighted knn regressor for k = 1; 2; 3; 5; 10; 20; 50; 100')
for k in k_list:
    p_test, weight = myLWR(train,test,k)
    err = np.linalg.norm(y_test - p_test)
    print('For k = ',str(k),' Error is ',err)
    print('For k = ',str(k),' Weights are ',weight)
    print('\n')
    
del train, test, y_test