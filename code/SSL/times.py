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
colors=['xkcd:apple','xkcd:grapefruit',
                                'xkcd:sky','olive',
                                'xkcd:muted blue','peru','tab:pink',
                                'deeppink', 'steelblue', 'tan', 'sienna',\
                                'olive', 'coral']
default_cycler = (cycler(color=colors))
tex_font = True
if tex_font:    
    rc('font',**{'family':'lmodern','serif':['Times'],'size':14})
    rc('text', usetex=True)
rc('lines', linewidth=2, linestyle='-')
rc('axes', prop_cycle=default_cycler)

#%%
d = 2
np.random.seed(422)
num_pts = 1000

#%%
s = np.linspace(0.001, 1, 5)
times = []
nnzs = []
epsilons=[]
ds = [2,10,20]
for d in ds:
    nnzs_loc = []
    epsilons_loc=[]
    X = np.random.uniform(0.,1., size=(num_pts,d))
    
    W = gl.weightmatrix.epsilon_ball(X,epsilon=d, kernel='distance')
    max_dist = W.max()
    epsilons_prop = np.linspace(0.001, max_dist, 15)
    
    for epsilon in epsilons_prop:
        try:
            W = gl.weightmatrix.epsilon_ball(X,epsilon=epsilon, kernel='distance')
            nnzs_loc.append(W.nnz)
            epsilons_loc.append(epsilon)
            print('finished for eps=' + str(epsilon) + ', d: ' + str(d))
        except:
            print('No connection for eps=' + str(epsilon) + ', d: ' + str(d))
        
    nnzs.append(nnzs_loc)
    epsilons.append(epsilons_loc)
#%%
print('Finished calc!')
#%%
plt.close('all')
fig, ax = plt.subplots(1,1)
ps=[]

for i,nnzs_loc in enumerate(nnzs):
    ax.plot(epsilons[i], nnzs_loc, label='d= ' + str(ds[i]), color=colors[i])
    ax.scatter(epsilons[i], nnzs_loc, color=colors[i])
    

ax.set_ylabel('Number of non-zero weights')
ax.set_xlabel(r'Scale $\varepsilon$')
ax.legend()

#%%
save = True
plt.tight_layout(pad=0.1)
fig.set_size_inches(8.5, 6.5)
if save:
    plt.savefig('NNZ.pdf')
