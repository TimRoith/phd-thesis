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
J = lambda x:x**2
theta = 0.3
ttheta = -1.2
DJ = lambda p:lambda x:2*x*p - J(p)

#%%
s = np.linspace(-2,2,40)
plt.close('all')

plt.plot(s, J(s), label='$J$')
plt.plot(s, DJ(theta)(s))
plt.plot(theta, J(theta), marker='o', markersize=10, color='xkcd:sky', label=r'$\theta$')
plt.plot(ttheta, J(ttheta), marker='o', markersize=10, color='tab:pink', label=r'$\overline{\theta}$')
plt.plot(ttheta, DJ(theta)(ttheta), marker='o', markersize=10, color='tab:pink')
plt.plot([ttheta, ttheta], [J(ttheta), DJ(theta)(ttheta)], color='tab:pink', 
         linestyle='dotted', label=r'$D_J^p(\overline{\theta},\theta)$')

plt.legend()
#%%
save = True
plt.tight_layout(pad=0.1)
if save:
    plt.savefig('lin_dist.pdf')
