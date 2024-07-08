import numpy as np
from mmica.solvers import gen_idx, compute_A_idx, compute_A, Huber, Sigmoid
from mmica._utils import python_cg_c, cython_cg_c, python_cg_c_with_extra_info


def min_W_with_extra_info(W, A, maxiter_cg, tol=1e-10):
    N, _ = W.shape
    hit_iters = np.empty(N)
    hit_norms = np.ones((N,maxiter_cg)) * np.nan
    for i in range(N):
        K = W @ A[i] @ W.T
        s, hit_iters[i], hit_norms[i,:] = python_cg_c_with_extra_info(B=K, i=i, max_iter=maxiter_cg, tol=tol, N=K.shape[0])
        s /= np.sqrt(s[i])
        W[i] = s @ W
    return W, hit_iters, hit_norms

class mmICA:
    def __init__(self, p, W_init=None, density='huber', maxiter_cg=10, greedy=0, alpha=.7, track_extra_info=False, tol=1e-10):
        self.p = p
        self.alpha = alpha
        self.greedy = greedy
        self.maxiter_cg = maxiter_cg
        self.tol = tol

        self.hit_iter_history = None
        self.hit_norm_history = None
        if track_extra_info:
            self.hit_iter_history = []
            self.hit_norm_history = []

        self.W = W_init.copy() if W_init is not None else np.eye(p)

        self.density = {'huber': Huber(),
                        'tanh': Sigmoid()}.get(density)

        self.A = np.zeros((p, p, p))
        self.n = 0

        self.cumulants = np.zeros(p)
        self.columns_seen = 0

    def observe_new_batch(self, x):
        _, batch_size = x.shape
        y = np.dot(self.W, x)
        u = self.density.ustar(y)
        step = 1. / (self.n + 1) ** self.alpha
        self.A *= (1 - step)
        if self.greedy:
            u *= step * self.p / self.greedy
            update_idx = gen_idx(self.p, self.greedy, batch_size)
            self.A += compute_A_idx(u, x, update_idx)
        else:
            u *= step
            self.A += compute_A(u, x)
        self.W, hit_iters, hit_norms = min_W_with_extra_info(W=self.W, A=self.A, maxiter_cg=self.maxiter_cg, tol=self.tol)
        if self.hit_iter_history is not None:
            self.hit_iter_history.append(hit_iters)
            self.hit_norm_history.append(hit_norms)

        # extra added by jgould
        non_gaussianness = self.density.logp(self.W @ x).mean(axis=1)
        self.cumulants = self.cumulants + (non_gaussianness - self.cumulants) / (self.columns_seen + batch_size)
        self.columns_seen += batch_size

        self.n += 1
