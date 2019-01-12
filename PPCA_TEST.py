from ppca import *
import numpy as np
import sklearn.datasets as dataset
import os
import matplotlib.pyplot as plt
# matplotlib inline
import matplotlib.cm as plt_cm
import matplotlib.colors as plt_col

def plot_scatter(x, classes, ax=None):
    ax = plt.gca() if ax is None else ax
    cmap = plt_cm.jet
    norm = plt_col.Normalize(vmin=np.min(classes), vmax=np.max(classes))
    mapper = plt_cm.ScalarMappable(cmap=cmap, norm=norm)
    colors = mapper.to_rgba(classes)
    #colors = mapper.to_rgba(range(50))
    ax.scatter(x[0, :], x[2, :], color=colors, s=10)

    plt.show()
    
    
def plot_digits(x, ax=None):
    xx = x[0, :]
    yy = x[1, :]
    width = np.max(xx) - np.min(xx)
    height = np.max(yy) - np.min(yy)
    ax = plt.gca() if ax is None else ax
    ax.set_xlim([np.min(xx) - 0.1 * width, np.max(xx) + 0.1 * width])
    ax.set_ylim([np.min(yy) - 0.1 * height, np.max(yy) + 0.1 * height])
    cmap = plt_cm.jet
    norm = plt_col.Normalize(vmin=0, vmax=39)
    mapper = plt_cm.ScalarMappable(cmap=cmap, norm=norm)
    colors = mapper.to_rgba(range(50))
    for digit, x in enumerate (zip(xx, yy)):
        ax.text(x[0], x[1], digit, color=colors[int(digit)])
        
    plt.show()
    
iris = dataset.load_iris()
iris_Y = np.transpose(iris.data)
iris_Y = np.transpose(iris.data)
iris_labels = iris.target

print(iris_Y)

print()
print()

print(iris_labels)
print()

plot_scatter(iris_Y, iris_labels)

# TOBOMOVIRUS 
data = np.loadtxt("tobomovirus.txt")
print()
print(data)
print(np.shape(data))
print()

# 
# # DIGITS DATA 
# 
# digits = dataset.load_digits()
# digits_i = np.random.choice(range(digits.data.shape[0]), 100)
# digits_y = np.transpose(digits.data[digits_i, :])
# digits_labels = digits.target[digits_i]
# 
# print()
# print(digits.data.shape[0])
# print(np.shape(digits.data))
# print(np.shape(digits_y))
# print(digits_y)
# 
# print()
# print()
# 
# print(digits_labels)
# print()
# 
# data_T = np.transpose(data)
# 
# # q -> Number of dimensions
# ppca = PPCA(q = 2)
# ppca.fit(data_T)
# 
# X = ppca.transform()
# print()
# print(X)
# 
# plot_digits(X)