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
num_pts = 100000
X = np.random.uniform(0.,1., size=(num_pts,2))

XXX = np.array([[0.2,0.5], [0.8,0.5]])
X = np.concatenate((XXX, X,))
#W = gl.weightmatrix.knn(X,10)

#%%
num_e = 15
U = np.zeros((X.shape[0], num_e))
UU = np.zeros((X.shape[0], num_e))
E = np.linspace(0.1,2.,num_e)

W = gl.weightmatrix.epsilon_ball(X,epsilon=0.1)
W = gl.weightmatrix.knn(X, 15)

train_ind = [0,1]
train_labels = np.array([[-1],[1]])

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
#%%
print('Finished calc!')
#%%
plt.close('all')
fig, ax = plt.subplots(1,2, subplot_kw={'projection': '3d'})

for i in range(2):
    ax[i].view_init(elev=20., azim=-120)

ax[0].set_title('Laplacian Learning $p=2$')
ax[1].set_title('Lipschitz Learning $p=\infty$')

ax[0].plot_trisurf(X[:,0], X[:,1], u[:,0], cmap='coolwarm')
ax[1].plot_trisurf(X[:,0], X[:,1], uu, cmap='coolwarm')

#%%
save = True
plt.tight_layout(pad=0.1)
fig.set_size_inches(8.5, 6.5)
if save:
    plt.savefig('2Dex_'+str(num_pts)+'.png')
