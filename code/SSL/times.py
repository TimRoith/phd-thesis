import numpy as np
import graphlearning as gl
import graphlearning.utils as utils
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import time

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
num_pts = 5000
X = np.random.uniform(0.,1., size=(num_pts,2))

X[0,0]=-1
X[1,0]=1
#W = gl.weightmatrix.knn(X,10)

#%%
epsilons = np.linspace(0.01,1., 5)
times = []
for epsilon in epsilons:
    t_start = time.perf_counter()
    
    W = gl.weightmatrix.epsilon_ball(X,epsilon=epsilon)
    #W = gl.weightmatrix.knn(X, 15)
    
    train_ind = [0,1]
    train_labels = np.array([[-1],[1]])

    #%%
    G = gl.graph(W)
    
    p = 2
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
    
    times.append(time.perf_counter() - t_start)
    print('finished for eps=' + str(epsilon) + ', time: ' + str(times[-1]))
    del G
#%%
print('Finished calc!')
#%%
plt.close('all')
fig, ax = plt.subplots()
ax.plot(epsilons, times)
ax.scatter(epsilons, times)
ax.set_ylabel('Computation Time [s]')
ax.set_xlabel(r'Scale $\varepsilon$')

#%%
save = True
plt.tight_layout(pad=0.1)
fig.set_size_inches(8.5, 6.5)
if save:
    plt.savefig('Times.png')
