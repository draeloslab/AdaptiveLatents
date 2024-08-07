import numpy as np
from .transformer import TypicalTransformer
from mmica.solvers import gen_idx, compute_A_idx, compute_A, Huber, Sigmoid, min_W
from mmica._utils import python_cg_c, cython_cg_c, python_cg_c_with_extra_info


def min_W_with_extra_info(W, A, maxiter_cg, tol=1e-10):
    N, _ = W.shape
    hit_iters = np.empty(N)
    hit_norms = np.ones((N, maxiter_cg)) * np.nan
    for i in range(N):
        K = W @ A[i] @ W.T
        s, hit_iters[i], hit_norms[i, :] = python_cg_c_with_extra_info(B=K, i=i, max_iter=maxiter_cg, tol=tol, N=K.shape[0])
        s /= np.sqrt(s[i])
        W[i] = s @ W
    return W, hit_iters, hit_norms


class BaseMMICA:
    def __init__(self, density='huber', maxiter_cg=10, greedy=0, alpha=.7, track_extra_info=False, tol=1e-10):
        self.alpha = alpha
        self.greedy = greedy
        self.maxiter_cg = maxiter_cg
        self.tol = tol

        self.density = {'huber': Huber(), 'tanh': Sigmoid()}.get(density)

        self.hit_iter_history = None
        self.hit_norm_history = None
        if track_extra_info:
            self.hit_iter_history = []
            self.hit_norm_history = []

        self.p = None
        self.W = None

        self.A = None
        self.n = 0

        self.cumulants = None
        self.columns_seen = 0

    def set_p(self, p):
        self.p = p

        self.W = np.eye(p)

        self.A = np.zeros((p, p, p))
        self.n = 0

        self.cumulants = np.zeros(p)
        self.columns_seen = 0

    def observe_new_batch(self, x):
        p, batch_size = x.shape
        if self.p is None:
            self.set_p(p)
        y = np.dot(self.W, x)
        u = self.density.ustar(y)
        step = 1. / (self.n + 1)**self.alpha
        self.A *= (1 - step)
        if self.greedy:
            u *= step * self.p / self.greedy
            update_idx = gen_idx(self.p, self.greedy, batch_size)
            self.A += compute_A_idx(u, x, update_idx)
        else:
            u *= step
            self.A += compute_A(u, x)
        self.W = min_W(self.W, self.A, self.maxiter_cg)
        # self.W, hit_iters, hit_norms = min_W_with_extra_info(W=self.W, A=self.A, maxiter_cg=self.maxiter_cg, tol=self.tol)
        # if self.hit_iter_history is not None:
        #     self.hit_iter_history.append(hit_iters)
        #     self.hit_norm_history.append(hit_norms)

        # extra added by jgould
        non_gaussianness = self.density.logp(self.W @ x).mean(axis=1)
        self.cumulants = self.cumulants + (non_gaussianness - self.cumulants) / (self.columns_seen + batch_size)
        self.columns_seen += batch_size

        self.n += 1

    def unmix(self, x):
        return self.W @ x


class mmICA(TypicalTransformer, BaseMMICA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processing_queue = []

    def pre_initialization_fit_for_X(self, X):
        self.set_p(X.shape[1])
        self.is_initialized = True
        self.partial_fit_for_X(X)

    def partial_fit_for_X(self, X):
        self.processing_queue.extend(X)
        if len(self.processing_queue) >= self.p:
            self.observe_new_batch(np.array(self.processing_queue).T)

    def transform_for_X(self, X):
        return self.unmix(X.T).T