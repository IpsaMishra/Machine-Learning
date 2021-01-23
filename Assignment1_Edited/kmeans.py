import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def generate_data(mu1,mu2,mu3,sigma,n_samples):
    x1 = np.random.multivariate_normal(mu1,sigma,n_samples)
    x2 = np.random.multivariate_normal(mu2,sigma,n_samples)
    x3 = np.random.multivariate_normal(mu3,sigma,n_samples)
    x = np.concatenate((x1,x2,x3),axis=0)
    return x

def mykmeans(X,k):
    n_iter = 200
    n_X = np.shape(X)[0]
    tags = np.zeros(n_X)
    
    iter_centers = []
    
    c = np.zeros((k,np.shape(X)[1]))    # Initialize the vector for the centers
    for i in range(0,k):
        c[i] = 5 * np.random.rand(1,np.shape(X)[1])     # Initialize the centers
        
    for nc in range(len(c)):
        iter_centers.append([])
        
    for nc in range(len(c)):
        iter_centers[nc].append(list(c[nc]))
        
    for i in range(n_iter):                     # Run a loop for all iterations
        for j in range(n_X):                    # Run a loop over all data points
            dist = [np.linalg.norm(X[j] - c[n]) for n in range(len(c))]
            indx = np.where(dist == np.amin(dist))
            indx = list(indx[0])
            tags[j] = indx[0]
            
        upd_c = c * 0.0
        for j in range(len(c)):                 # Update each of the centroids
            identify = (tags == j) + 0.0        # Marks 1 for the data which is part of cluster 'j'
            upd_c[j] = np.matmul(np.transpose(identify) , X) / np.sum(identify)
        
        c = upd_c
        for nc in range(len(c)):
            iter_centers[nc].append(list(c[nc]))
        
    iter_centers = np.array(iter_centers)
    
    for nc in range(len(c)):                    # Plots the values of center updates
        plt.scatter(iter_centers[nc,:,0],iter_centers[nc,:,1],marker = '.')
    
    print('# of iterations = ', len(iter_centers[0])-1)              # Prints the number of iterations
    
    return c, tags

##Generate Dataset
mu1 = np.array([-3,0])     
mu2 = np.array([3,0])
mu3 = np.array([0,3])
sigma = np.array([[1,0.75],[0.75,1]])
n_samples = 300
X = generate_data(mu1, mu2, mu3, sigma, n_samples)

c, tags = mykmeans(X,2)
print(c)
