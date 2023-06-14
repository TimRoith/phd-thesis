import numpy as np
import graphlearning as gl
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import optimizers as opt

from proj_l1	 import euclidean_proj_l1ball

#%%
class linf_pd:
    '''
    Linfty primal dual
    '''
    def __init__(self, G, f, ind, sigma=0.1, tau=0.1, theta=1., max_it=1000, tol=1e-6, lamda=1.):
        self.G = G
        self.f = f
        self.ind = ind
        self.u = 0.5*np.ones(G.num_nodes)
        #self.u = np.random.uniform(0,1,G.num_nodes)
        self.u_bar = np.zeros(G.num_nodes)
        self.q = G.gradient(self.u_bar, weighted=True)
        self.sigma = sigma
        self.tau = tau
        self.theta = theta
        self.max_it = max_it
        self.num_it = 0
        self.tol = tol
        self.term = False
        self.lamda = lamda

    def step(self):
        self.u_old = self.u
        self.q = self.proj(self.q + self.sigma * self.G.gradient(self.u_bar, weighted=True), self.sigma)
        self.u = self.proxG(self.u + self.tau * self.G.divergence(self.q, weighted=True), self.tau)
        #self.u = 1/(1+self.tau) * (self.u + self.tau * self.G.divergence(self.q, weighted=True)) + self.tau/(1+self.tau) * self.f
        self.u_bar = self.u + self.theta * (self.u - self.u_old)

        self.num_it += 1
        self.check_term()

        print('Itertion: ' + str(self.num_it) + ' energy: ' + str(self.energy()))

    def proxG(self, u, tau):
        u[self.ind] = self.f[self.ind]
        return u

    def proj(self, u, sigma):
        u.data = euclidean_proj_l1ball(u.data, s=sigma)
        return u
    
    def energy(self):
        return np.linalg.norm(self.G.gradient(self.u, weighted=True).data, ord=float('inf'))

    def check_term(self):
        if self.num_it >= self.max_it:
            self.term = True
        elif np.linalg.norm(self.u - self.u_old) < self.tol:
            self.term = True

class linf(gl.ssl.ssl):
    def __init__(self, W, class_priors=None, sigma=0.1, tau=0.1, theta=1., max_it=1000, lamda=1.):
        super().__init__(W, class_priors)
        self.sigma = sigma
        self.tau = tau
        self.theta = theta
        self.max_it = max_it
        self.onevsrest = True
        self.lamda = lamda

    
    def _fit(self, train_ind, train_labels, all_labels=None):
        f = np.zeros(self.graph.num_nodes)
        f[train_ind] = train_labels

        class gradient:
            def __init__(self, graph):
                self.graph = graph

            def __call__(self, u):
                return self.graph.gradient(u, weighted=True)

            def adjoint(self, q):
                return -self.graph.divergence(q, weighted=True)
            
        A = gradient(self.graph)

        class fstar:
            def prox(u, sigma):
                u.data = euclidean_proj_l1ball(u.data, s=sigma/self.lamda)
                return u
            
        class fstar:
            def prox(u, sigma):
                z = u.data
                #z[np.abs(z) < sigma/self.lamda] = 0.
                u.data = np.clip(z, -sigma/self.lamda, sigma/self.lamda) / u.nnz
                return u
            
        # class fstar:
        #     def prox(u, sigma):
        #         return 1/(1+self.sigma) * u
            
        class g:
            def prox(u, tau):
                u[train_ind] = f[train_ind]
                return u
            
        def energy_fun(u):
            return np.linalg.norm(self.graph.gradient(u, weighted=True).data, ord=float('inf'))
        
        def compute_primal_res(pd):
            pp = 1/pd.tau * (pd.u_old - pd.u) + pd.theta * A.adjoint(pd.p_old - pd.p)
            return np.linalg.norm(pp)

        def compute_dual_res(pd):
            dd = 1/pd.sigma * (pd.p_old - pd.p) - pd.theta * A(pd.u_old - pd.u)
            return np.linalg.norm(dd.data)
        
        u = 0.0*np.ones(self.graph.num_nodes)
        p = self.graph.gradient(u, weighted=True)

        self.pd = opt.pdgh(
            A, fstar, g, u, p,
            max_it=self.max_it,
            theta=self.theta,
            tau=self.tau, sigma=self.sigma,
            energy_fun=energy_fun,
            verbosity=1,
            compute_primal_res=compute_primal_res,
            compute_dual_res=compute_dual_res,)
        
        def term_crit(pd,tol=1e-16, tol_energy=1e-9):
            if pd.num_it > self.max_it:
                return True
            # if hasattr(pd, 'dual_res') and hasattr(pd, 'primal_res'):
            #     if (pd.primal_res**2 + pd.dual_res**2) < tol:
            #         return True
                
            if pd.num_it > 10 and (abs(pd.energy_hist[-1] - pd.energy_hist[-2]) < tol_energy):
                return True
            return False
        
        def compute_B(pd, c=0.9):
            p_diff = (pd.p - pd.p_old)
            B = c/(2 * pd.tau)   *  np.linalg.norm(pd.u - pd.u_old)**2 +\
                c/(2 * pd.sigma) * np.linalg.norm(p_diff.data)**2 -\
                2 * np.vdot(A.adjoint(p_diff).ravel(), ((pd.u - pd.u_old)).ravel())
            B = B.real
            return B
        
        scheduler = opt.adaptive_pd_stepsize(self.pd, compute_B=compute_B, verbosity=1, gamma=0.95, s=0.01)

        #self.pd = linf_pd(self.graph, f, train_ind, sigma=self.sigma, tau=self.tau, theta=1., max_it=self.max_it, tol=1e-9, lamda=1.)
        while not term_crit(self.pd):
            self.pd.step()
            #scheduler()

        return self.pd.u


#%%
X,labels = datasets.make_moons(n_samples=500,noise=0.1)
W = gl.weightmatrix.knn(X,10)

train_ind = gl.trainsets.generate(labels, rate=6)
train_labels = labels[train_ind]

#model = gl.ssl.laplace(W)
model_pd = linf(W, sigma=.005, tau=111110.1, theta=1.,
                max_it=1000, lamda=21.1)
#model = gl.ssl.amle(W)
#pred_labels = model.fit_predict(train_ind, train_labels)

#%%
u = model_pd.fit(train_ind, train_labels)
u = u[:,0] - u[:,1]
u = np.clip(u,-1.,1.)
#%%
model = gl.ssl.amle(W, weighted=True)
v = model.fit(train_ind, train_labels)
v = v[:,0] - v[:,1]

#accuracy = gl.ssl.ssl_accuracy(pred_labels, labels, len(train_ind))   
#print("Accuracy: %.2f%%"%accuracy)

#%%
plt.close('all')
plt.scatter(X[:,0],X[:,1], c=u, cmap='seismic')
plt.scatter(X[train_ind,0],X[train_ind,1], c='g')

plt.figure()
plt.scatter(X[:,0],X[:,1], c=v, cmap='seismic')
plt.scatter(X[train_ind,0],X[train_ind,1], c='g')
plt.title('AMLE')
plt.show()