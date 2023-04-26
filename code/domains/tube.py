import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
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
def polar2cart(x):
    xx = x[:,0]*np.cos(x[:,1])
    yy = x[:,0]*np.sin(x[:,1])
    return np.stack([xx,yy]).T

def sphere(pos, r=1., theta_1=0., theta_2=2*np.pi, rot = 0., num_pts = 200):
    thetas = np.linspace(theta_1, theta_2, num_pts)[1:-1,None] + rot
    sphere = np.concatenate([r*np.ones((num_pts-2,1)), thetas], axis=1)
    return polar2cart(sphere) + pos

def upper_curve(t, phi=3/18 * np.pi):
    e = polar2cart(np.array([[2, phi]]))
    ee = 1+e[0,0]
    if t < 3*np.cos(phi):
        return np.sqrt(9 - t**2)
    else:
        return np.sqrt(1 - (t-e[0,0])**2) + e[0,1]
    


num_pts = 435
phi = (30/180) * np.pi
thetas = np.linspace(phi,2*np.pi-phi, num_pts)[:,None]


inner = polar2cart(np.concatenate([np.ones((num_pts,1)), thetas], axis=1))
outer = polar2cart(np.concatenate([3*np.ones((num_pts,1)), thetas], axis=1))
outer = np.flip(outer, axis=0)
d = polar2cart(np.array([[2, -phi]]))
lower_cap = sphere(d, r=1, theta_1=0, theta_2=np.pi, rot=-phi)
lower_cap = np.flip(lower_cap, axis=0)
e = polar2cart(np.array([[2, phi]]))
upper_cap = sphere(e, r=1, theta_1=0, theta_2=np.pi, rot=np.pi+phi)
upper_cap = np.flip(upper_cap, axis=0)

c = np.concatenate([inner, lower_cap, outer, upper_cap], axis=0)

#%%
t = np.linspace(0,1, c.shape[0])
cs = scp.interpolate.CubicSpline(t, c)

lb = np.concatenate([inner, upper_cap], axis=0)
lb = lb[np.where(lb[:,1] >0)]
lb = lb[np.where(lb[:,1] <1)]
idx = np.argsort(lb[:,0])
lb = lb[idx,:]
add =  np.stack([np.linspace(-3,-1, 150), np.zeros((150))]).T
lb = np.concatenate((add, lb), axis=0)
ub = np.array([upper_curve(lb[i,0]) for i in range(lb.shape[0])])
#%%
path = np.concatenate((e, np.array([[0,1]])), axis=0)
path = np.concatenate((path, sphere(np.zeros((1,2)), 
                                    theta_1=0., theta_2=np.pi,
                                    rot=np.pi/2,r=1.0)), axis=0)
path = np.concatenate((path, d), axis=0)


#%%
plt.close('all')
#plt.scatter(c[:,0], c[:,1])
ccs = cs(t)
plt.plot(ccs[:,0], ccs[:,1], linewidth=3, color='xkcd:grapefruit', label='$\partial\Omega$')
plt.fill_between(lb[:,0], lb[:,1], ub, color='xkcd:sky',alpha=0.5, label='$\Omega$')
plt.fill_between(lb[:,0], -ub, -lb[:,1], color='xkcd:sky',alpha=0.5)

plt.scatter(e[0,0], e[0,1], label='x')
plt.scatter(d[0,0], d[0,1], label='y')

plt.plot(path[:,0], path[:,1], color='xkcd:apple',alpha=1, linestyle='dotted')
plt.plot([e[0,0],d[0,0]], [e[0,1],d[0,1]], color='xkcd:muted blue',alpha=1)

plt.axis('equal')
plt.legend()

#%%
plt.tight_layout(pad=0.1)
plt.savefig('tube.pdf')
    
