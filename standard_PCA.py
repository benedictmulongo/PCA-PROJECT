from scipy import linalg
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import matplotlib.colors as plt_col
from matplotlib import cm
import json
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


class PCA(object):
    def __init__(self, X,K = 2):
        """

        X : with N observations and D dimensions
        K : The number of principal components wanted by the user

        
        """
        N,D = np.shape(X)
        self.N = N
        self.D = D
        self.data = X
        self.K = K
        self.proj_data = np.zeros((N,K))
        self.W = None 
        self.reconst = np.zeros((N,D))
        self.mu = 0

        
    def fit(self) :
        
        X = self.data
        N,D = np.shape(X)
        
        
        # u = sum(x_i)/N
        mu = X.mean(0)
        self.mu = mu
        
        # X = [x_1 - u, x_2 - u, ..., x_N - u, ]
        Xm = X - np.matlib.repmat(mu,N,1)
        
    
        data = np.matmul(Xm.T, Xm) 
        U,S,V = linalg.svd(data, full_matrices=True)
        
        W = np.matmul(U, np.diag(np.sqrt(S)) )
        self.W = W
    
        
        for i, x_i in enumerate(X) :
            self.proj_data[i,:] = np.matmul(W[:,0:self.K].T , Xm[i,:])
            
        self.proj_data /= 100
            
        return self.proj_data, W[:,0:self.K].T, mu 
        
    def reconstruct_data(self) :
        
        for i in range(self.N) :
            self.reconst[i,:] = self.mu + np.matmul(self.W[:,0:self.K] , self.proj_data[i,:])
            
        return self.reconst
        
    def plot_digits(self, ax=None):
        
        x = self.proj_data
        xx = x[:,0]
        yy = x[:,1]
        
        
        width = np.max(xx) - np.min(xx)
        height = np.max(yy) - np.min(yy)
        ax = plt.gca() if ax is None else ax
        ax.set_xlim([np.min(xx) - 0.1 * width, np.max(xx) + 0.1 * width])
        ax.set_ylim([np.min(yy) - 0.1 * height, np.max(yy) + 0.1 * height])
        
        for digit, x in enumerate (zip(xx, yy)):
            ax.text(x[0], x[1], digit+10, color='k')
            
        
        plt.show()
        
def test() :
    data = np.loadtxt("tobomovirus.txt")
    # data = [[1,2 ,3 ,5],[ 5, 6 ,9 ,10],[ -2, 3, 4, 11]]
    # data = np.array(data)
    pca_model = PCA(data, K = 2 )
    x_proj, W, mu  = pca_model.fit()
    pca_model.plot_digits()
    # data_prim  = pca_model.reconstruct_data()
    # pca_model.plot_digits()
    # print(x_proj)
    # print()
    # print(W)
    # print()
    # print(mu)
    # print("data_prim")
    # print(data_prim)
    
def find(dir = 'C:/Users/ben/Desktop/BachelorProject/dico.json', key = 'wash'):
    f = open(dir)
    filen = json.load(f)
    f.close()
    print(filen[key])
    print('sample' in filen[key]['word'])
    
def fetch(data = '2017'):
    dir = 'countries_data_' + str(data)+'_scores_int.json'
    
    f = open(dir)
    filen = json.load(f)
    f.close()
    
    return filen['names'], filen['data'], filen['readme']
    
def plot_data(x, country_name, ax=None):

    xx = x[:,0]
    yy = x[:,1]
    
    plt.scatter(xx, yy, c=target, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('spectral', 7))
    plt.xlabel('$W_1$')
    plt.ylabel('$W_2$')
    plt.colorbar()

    ax = plt.gca() if ax is None else ax

    
    for digit, x in enumerate (zip(xx, yy)):
        ax.text(x[0], x[1], country_name[digit], color='k',fontsize=8)
        
    
    plt.show()

country_name, country_data, readme = fetch('2017')

country_data = np.array(country_data)
N,D = np.shape(country_data)

print(country_name)
print()
print(country_data)
print()
print(country_data[:,2:])


# from sklearn.decomposition import PCA

target = country_data[:,1].reshape((1,N)).flatten()
print("target = ", target)



# pca = PCA(n_components=2)
# proj = pca.fit_transform(country_data[:,2:])
pca_model = PCA(country_data, K = 2 )
proj, W, mu  = pca_model.fit()
print()
print(proj)

plot_data(proj,country_name)





