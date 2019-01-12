import seaborn as sns
import numpy as np 
import numpy.matlib
import numpy as np
import sklearn.datasets as ds
import os
import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import matplotlib.colors as plt_col
from matplotlib import cm


NotNum = np.nan
data = np.loadtxt("tobomovirus.txt")
data = [[1,2 ,3 ],[ 5, 6 ,9 ],[ -2, 3, 4]]
data = [[1,2 ,3 ,5],[ 5, 6 ,9 ,10],[ -2, 3, 4, 11]]
data = [[1,2 ,NotNum ,5],[ 5, 6 ,9 ,10],[ -2, NotNum, 4, 11]]


data = [[-0,-0,NotNum,5,0],[ -0,NotNum,-0, 0 , 0],[-2, 0,-1,1,2]]
    
data = np.array(data)


# N = number of data, D = dimension
N,D = np.shape(data)
# Latent space dimensionality 
d = 2
threshold = 1e-4

def clean_data(data):
    NotNum = np.nan
    hidden = np.isnan(data)
    missing = np.sum(hidden)
    
    data_clean = np.copy(data)
    
    data_clean[hidden] = 0
    
    mu = np.mean(data_clean,0)

    return data, data_clean,mu, hidden, missing
    
data, data_clean,mu, hidden, missing = clean_data(data)

print(data)
print()
print(data_clean)
print()
print(mu)
print()

C = np.random.randn(D,d)
CtC = np.matmul(C.T,C)

X1 = np.matmul( data_clean, C )
X = np.matmul( X1 , np.linalg.inv(CtC))
recon = np.matmul( X , C.T )
recon[hidden] = 0

s = recon - data_clean
ss = np.sum(np.sum(s**2)) / (N*D - missing)

print()
print(np.sum(s**2))
print(ss)

count = 1
old = np.inf
 