import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from cycler import cycler
#%%
g_1 = lambda x: 0 * x
g_2 = lambda x: -np.abs(x-0.5) + 0.5
g_3 = lambda x: np.abs(x-0.5) - 0.5

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
x = np.linspace(-1,1,520)
o = np.array([-1,0,1])

plt.scatter(o, g_1(o), label='$g_1$')
plt.scatter(o, g_2(o), label='$g_2$')
plt.scatter(o, g_3(o), label='$g_3$')
plt.plot(x,g_1(x), label='$\overline{g_1}$')
plt.plot(x,g_2(x), label=r'$\overline{g_2}$')
plt.plot(x,g_3(x), label=r'$\underline{g_3}$')

plt.legend()
#%%
plt.tight_layout(pad=0.1)
plt.savefig('comp.pdf')
