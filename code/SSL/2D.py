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
num_pts = 10000
X = np.random.uniform(0.,1., size=(num_pts,2))

XXX = np.array([[0.2,0.5], [0.8,0.5]])
X = np.concatenate((XXX, X,))
#W = gl.weightmatrix.knn(X,10)

#%%
epsilon = 1.2*(np.log(num_pts)/num_pts)**(0.5)

W = gl.weightmatrix.epsilon_ball(X,epsilon=epsilon)
#W = gl.weightmatrix.knn(X, 15)

train_ind = [0,1]
train_labels = np.array([[-1],[1]])

#%%
G = gl.graph(W)

p = 'inf'
if p==2:
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
    u = u[:,0]
else:
    u = G.amle(train_ind, train_labels, tol=1e-10)
    

#%%
print('Finished calc!')
#%%
plt.close('all')
fig, ax = plt.subplots(1,1, squeeze=False,subplot_kw={'projection': '3d'})

for i in range(1):
    ax[i,0].view_init(elev=20., azim=-120)

ax[0,0].set_title('$n=$' + str(num_pts))
#ax[1].set_title('Lipschitz Learning $p=\infty$')

ax[0,0].plot_trisurf(X[:,0], X[:,1], u, cmap='coolwarm')
#ax[1].plot_trisurf(X[:,0], X[:,1], uu, cmap='coolwarm')

#%%
save = True
plt.tight_layout(pad=0.1)
fig.set_size_inches(8.5, 6.5)
if save:
    plt.savefig('2Dex_'+str(num_pts)+ '-p='+str(p)+ '.png')
