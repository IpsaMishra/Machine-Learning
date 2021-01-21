import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from mpl_toolkits import mplot3d

  
def norm_data_generate(m, c, k = 1, bins = 10):      # Generates normally distributed data with given mean and variance/covariance
    if isinstance(m,float) == 1 or isinstance(m,int) == 1 or len(m) == 1:  # Convert into proper array format for use with numpy
        m = np.array([float(np.array([m]))])
        c = np.array([[float(np.array([c]))]])
        
    x = np.random.multivariate_normal(m,c,k)
    if bins == -1:          # Suppress the histogram plot
        return x
    elif len(m) == 1:
        plt.figure()
        plt.hist(x,bins)
    elif len(m) == 2:
        plt.figure()
        h = plt.hist2d(x[:,0], x[:,1])
        plt.colorbar(h[3])
    return x


def Gauss_mixt_data_generate(m0, c0, k0, m1, c1, k1, bins, scat):    # Generates data from a bi-modal Gaussian mixture
    x0 = norm_data_generate(m0, c0, k0, -1)
    x1 = norm_data_generate(m1, c1, k1, -1)
    x = np.concatenate((x0, x1))
    
    if isinstance(m0,float) == 1 or isinstance(m0,int) == 1 or len(m0) == 1:  # Convert into proper array format for use with numpy
        m = np.array([float(np.array([m0]))])
    else:
        m = m0
        
    if scat != 0:
        plt.figure()
        ax = plt.axes()
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_title('Scatter Plot')
        col = np.concatenate((2 * np.ones(k0), 3 * np.ones(k1)))
        plt.scatter(x[:,0], x[:,1], c = col, alpha = 0.5)
        
    if bins == -1:          # Suppress the histogram plot
        return x
    elif len(m) == 1:
        plt.figure()
        plt.hist(x,bins)
    elif len(m) == 2:
        plt.figure()
        h = plt.hist2d(x[:,0], x[:,1])
        plt.colorbar(h[3])
    return x

    
def label_gen(k0, k1):                                              # Generates binary classification labels for datasets
    l0 = np.zeros(k0)
    l1 = np.ones(k1)
    l = np.concatenate((l0, l1))
    return l


def mykde(x,h = 0.5):                   # Generates kernel density estimation values for a given bandwidth and plots it for 1D, 2D
    x = np.array(x)
    dims = np.shape(x)
    dmin = np.zeros((1,dims[1]))
    dmax = np.zeros((1,dims[1]))
    samp_rate = 8
    step = []
    for i in range(dims[1]):
        dmin[0,i] = np.min(x[:,i]) - 2 * h
        dmax[0,i] = np.max(x[:,i]) + 2 * h
        naxis = int((dmax[0,i] - dmin[0,i]) * samp_rate / h)   # Number of points along of axis where the value is computed
        step.append((dmax[0,i] - dmin[0,i]) / naxis)
        print('The domain of axis', i, '=', [dmin[0,i],dmax[0,i]])
    
    if dims[1] == 1:
        ax1 = dmin[0,i] + step[0] * np.arange(naxis + 1)
        vals = np.zeros(np.shape(ax1))
        for x_i in x:
            rv = ss.multivariate_normal(x_i, h*h)
            vals += rv.pdf(ax1) / len(x)
        
        plt.figure()
        ax = plt.axes()
        ax.set_xlabel('x')
        ax.set_ylabel('PDF')
        ax.set_title('Surface plot of a Gaussian KDE')
        plt.plot(ax1,vals)
    
    elif dims[1] == 2:
        ax1, ax2 = np.mgrid[dmin[0,0]:dmax[0,0]+0.01:step[0], dmin[0,1]:dmax[0,1]+0.01:step[1]]
        vals = np.zeros(np.shape(ax1))
        coord = np.empty(ax1.shape + (2,))
        for x_i in x:
            rv = ss.multivariate_normal(x_i, [[h, 0], [0, h]])
            coord[:, :, 0] = ax1 
            coord[:, :, 1] = ax2
            vals += rv.pdf(coord) / len(x)
        
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(ax1, ax2, vals, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('PDF')
        ax.set_title('Surface plot of a 2D Gaussian KDE')
        fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
        ax.view_init(60, 35)
        
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(ax1, ax2, vals)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('PDF')
        ax.set_title('Wireframe plot of a 2D Gaussian KDE');
   
    else:
        print('This function is defined for 1D, 2D datasets only')
        return
