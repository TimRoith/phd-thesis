import numpy as np
import graphlearning as gl
import graphlearning.utils as utils
import sklearn.datasets as datasets
import matplotlib.pyplot as plt

from scipy import sparse
from matplotlib.collections import LineCollection
from matplotlib import rc
from cycler import cycler
#%%
plt.close('all')
#plt.style.use(['ggplot'])
plt.style.use(['seaborn-whitegrid'])
default_cycler = (cycler(color=['xkcd:apple','xkcd:grapefruit',
                                'xkcd:sky','olive',
                                'xkcd:muted blue','peru','tab:pink',
                                'deeppink', 'steelblue', 'tan', 'sienna',\
                                'olive', 'coral']))
tex_font = True
if tex_font:    
    rc('font',**{'family':'lmodern','serif':['Times'],'size':14})
    rc('text', usetex=True)
rc('lines', linewidth=2, linestyle='-')
rc('axes', prop_cycle=default_cycler)

#%%
np.random.seed(422)
num_pts = 100
X = np.random.normal(loc=0.2,scale=.2, size=(num_pts,1))
XX = np.random.normal(loc=.8,scale=.2, size=(num_pts,1))
X = np.concatenate((X, XX))

XXX = np.array([[0.2], [0.8]])
X = np.concatenate((XXX, X,))
#W = gl.weightmatrix.knn(X,10)

#%%
num_e = 15
U = np.zeros((X.shape[0], num_e))
UU = np.zeros((X.shape[0], num_e))
E = np.linspace(0.1,2.,num_e)

for i,e in enumerate(E):
    W = gl.weightmatrix.epsilon_ball(X,epsilon=e)
    
    train_ind = [0,1]
    train_labels = np.array([[-1],[1]])
    
    #%%
    model = gl.ssl.laplace(W)
    
    #%%
    G = gl.graph(W)
    L = G.laplacian()
    n = G.num_nodes
    idx = np.full((n,), True, dtype=bool)
    idx[train_ind] = False
    
    #Right hand side
    b = -L[:,train_ind]*train_labels
    b = b[idx,:]
    
    #Left hand side matrix
    A = L[idx,:]
    A = A[:,idx]
    
    #Preconditioner
    m = A.shape[0]
    M = A.diagonal()
    M = sparse.spdiags(1/np.sqrt(M+1e-10),0,m,m).tocsr()
       
    #Conjugate gradient solver
    v = utils.conjgrad(M*A*M, M*b)
    v = M*v
    
    #Add labels back into array
    u = np.zeros((n,1))
    u[idx,:] = v
    u[train_ind,:] = train_labels
    uu = G.amle(train_ind, train_labels, tol=1e-10)
    
    U[:,i] = u[:,0]
    UU[:,i] = uu
#%%
plt.close('all')
idx = np.argsort(X[:,0])
XX = X[idx,0]
US = U[idx, :]
UUS = UU[idx, :]
EE, XM = np.meshgrid(E, XX)
fig, ax = plt.subplots(1,2, subplot_kw={'projection': '3d'})

for i in range(2):
    ax[i].set_box_aspect((np.ptp(XX), 3*np.ptp(E), np.ptp(US)))
    ax[i].view_init(elev=20., azim=230.)
    ax[i].set_ylabel('$h$')
    ax[i].set_xlabel('$x$')
    
ax[0].set_title('Laplacian Learning $p=2$')
ax[1].set_title('Lipschitz Learning $p=\infty$')
    
for i in range(num_e):
    ax[0].plot(XX, US[:,i], color='xkcd:grapefruit', alpha=1-0.05*i, zs=E[i], zdir='y')
    ax[1].plot(XX, UUS[:,i], color='xkcd:apple', alpha=1-0.05*i, zs=E[i], zdir='y')

save = False
if save:
    plt.tight_layout(pad=0.2)
    fig.set_size_inches(8.5, 6.5)
    plt.savefig('scaling.pdf')

#%%
plt.figure()
plt.contourf(XM, EE, U)


#%%
plt.scatter(X,u)
plt.scatter(X,uu, color='g')