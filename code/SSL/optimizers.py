class pdgh:
    def __init__(self,
        K, fstar, g, u, p,
        max_it=50,
        theta=1.0,
        tau=0.1, sigma=0.1, 
        energy_fun=None,
        verbosity=0,
        compute_primal_res = None,
        compute_dual_res = None,
        ):
        
        self.K = K
        self.fstar = fstar
        self.g = g
        self.u = u.copy()
        self.u_bar = u.copy()
        self.u_old = None
        self.p = p.copy()
        self.p_old = None
        self.max_it = max_it
        self.theta = theta
        self.tau = tau
        self.sigma = sigma
        self.energy_fun = energy_fun
        if not callable(energy_fun):
            self.energy_fun = lambda u: 0.

        self.verbosity = verbosity
        self.energy_hist = []
        self.num_it = 0

        if compute_primal_res is None:
            self.compute_primal_res = lambda: 0.
        else:
            self.compute_primal_res = compute_primal_res

        if compute_dual_res is None:
            self.compute_dual_res = lambda: 0.
        else:
            self.compute_dual_res = compute_dual_res

    def step(self):
        self.u_old = self.u
        self.p_old = self.p

        self.p = self.fstar.prox(self.p + self.sigma * self.K(self.u_bar), self.sigma)
        self.u = self.g.prox(self.u - self.tau * (self.K.adjoint(self.p)), self.tau)
        self.u_bar = self.u + self.theta * (self.u - self.u_old)

        if self.num_it%1==0:
            energy = self.energy_fun(self.u)

        self.energy_hist.append(energy)
        if self.verbosity > 0:
            print('Iteration: ' + str(self.num_it) + ', energy: ' + str(energy))

        # compute primal and dual residuals
        self.primal_res = self.compute_primal_res(self)
        self.dual_res   = self.compute_dual_res(self)

        
        self.num_it += 1

class adaptive_pd_stepsize:
    def __init__(
            self, pd, eta=0.95, gamma=0.95,
            s = 2.0,
            compute_B=None,
            verbosity=0):
        self.pd = pd
        self.eta = eta
        self.gamma = gamma
        self.verbosity = verbosity
        self.s = s

        if compute_B is None:
            self.compute_B = lambda: 0.
        else:
            self.compute_B = compute_B

    def __call__(self):
        B = self.compute_B(self.pd)

        if B <= 0:
            self.pd.tau = self.pd.tau/2
            self.pd.sigma = self.pd.sigma/2
        if self.pd.primal_res >= self.s * self.pd.dual_res:
            self.pd.tau *= 1/(1-self.gamma)
            self.pd.sigma *= (1-self.gamma)
            self.gamma *= self.eta
        elif self.pd.dual_res >= self.s * self.pd.primal_res:
            self.pd.tau *= (1-self.gamma)
            self.pd.sigma *= 1/(1-self.gamma)
            self.gamma *= self.eta

        if self.verbosity > 0:
            print('tau = ' + str(round(self.pd.tau, 3)) + ', sigma = ' + str(round(self.pd.sigma, 3)))