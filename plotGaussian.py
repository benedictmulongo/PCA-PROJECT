
# Code from Chapter 2 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Plots a 1D Gaussian function
# import pylab as pl
# import numpy as np
# 
# gaussian = lambda x: 1/(np.sqrt(2*np.pi)*1.5)*np.exp(-(x-0)**2/(2*(1.5**2)))
# x = np.arange(-5,5,0.01)
# y = gaussian(x)
# pl.ion()
# pl.plot(x,y,'k',linewidth=3)
# pl.xlabel('x')
# pl.ylabel('y(x)')
# pl.axis([-5,5,0,0.3])
# pl.title('Gaussian Function (mean 0, standard deviation 1.5)')
# pl.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.io

## 2D gaussian
# 
# def gaussian2d(mu,sigma,xvals,yvals):
#     const = (1.0/(2.0*np.pi))*(1.0/np.sqrt(np.linalg.det(sigma)))
#     si = np.linalg.inv(sigma)
#     xv = xvals-mu[0]
#     yv = yvals-mu[1]
#     return const * np.exp(-0.5*(xv*xv*si[0][0] + xv*yv*si[1][0] + yv*xv*si[0][1] + yv*yv*si[1][1]))
# 
# 
# xp = np.arange(-30,30,0.1)
# yp = np.arange(-10,10,0.1)
# Xp,Yp = np.meshgrid(xp,yp)
# Z = gaussian2d(prior_mean,prior_cov,Xp,Yp)
# CS = plt.contour(Xp,Yp,Z,20,colors='k')
# plt.xlabel('$w_0$')
# plt.ylabel('$w_1$')

## 3d


# 
# from mpl_toolkits import mplot3d
# import numpy as np
# import matplotlib.pyplot as plt
# 
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# 
# 
# 
# ax = plt.axes(projection='3d')
# 
# # Data for a three-dimensional line
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
# ax.plot3D(xline, yline, zline, 'gray')
# 

# # Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

##

# Mean vector and covariance matrix
mu = np.array([0., 0])
Sigma = np.array([[ 1. , 0], [0,  1]])



def multivariate_gaussian(mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """
    # Our 2-dimensional distribution will be over variables X and Y
    N = 100
    X = np.linspace(-5, 5, N)
    Y = np.linspace(-5, 5, N)
    X, Y = np.meshgrid(X, Y)
        
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    
    Z =  np.exp(-fac / 1) / N

    
    # The distribution on the variables X, Y packed into pos.
    
    plt.xlabel('$latentDim_0$', fontsize=8)
    plt.ylabel('$latentDim_1$', fontsize=8)
    # Create a surface plot and projected filled contour plot under it.
    plt.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.hot)


    
    plt.show()
    
# multivariate_gaussian(mu, Sigma)

def plot_2patterns(A1,A2, row, col) :
    
    A1 = np.array(A1)
    A2 = np.array(A2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].matshow(A1.reshape(row, col), cmap='bwr')
    ax[0].set_title('Corrupted pattern')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    
    ax[1].matshow(A2.reshape(row, col), cmap='bwr')
    ax[1].set_title('Recovered pattern')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    
    plt.show()

def plt_img_matrix(X, row, col):
    N = X.shape[0]
    X = X.reshape(row, col).transpose()

    plt.xticks([])
    plt.yticks([])
    plt.title("Image")
    plt.imshow(X, cmap=plt.get_cmap('gray'))

    plt.show()


def fetchDataset():
    from numpy import genfromtxt
    X = genfromtxt('olivettifacesX.txt', delimiter=',')
    X = X/255
    y = genfromtxt('olivettifacesY.txt', delimiter=',',dtype=np.int)
    pcadim = 20
    
    return X,y,pcadim
    

mat = scipy.io.loadmat('justturn.mat')
print(mat)