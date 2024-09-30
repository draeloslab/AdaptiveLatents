import numpy as np
from .transformer import TypicalTransformer
from mmica.solvers import gen_idx, compute_A_idx, compute_A, Huber, Sigmoid, min_W


class BaseMMICA:
    def __init__(self, density_name='huber', maxiter_cg=10, greedy=0, alpha=.7, track_extra_info=False, tol=1e-10):
        self.alpha = alpha
        self.greedy = greedy
        self.maxiter_cg = maxiter_cg
        self.tol = tol
        self.density_name = density_name

        self.density = {'huber': Huber(), 'tanh': Sigmoid()}.get(self.density_name)

        self.track_extra_info = track_extra_info
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

        # extra added by jgould
        non_gaussianness = self.density.logp(self.W @ x).mean(axis=1)
        self.cumulants = self.cumulants + (non_gaussianness - self.cumulants) / (self.columns_seen + batch_size)
        self.columns_seen += batch_size

        self.n += 1

    def unmix(self, x):
        return self.W @ x

    def remix(self, x):
        return np.linalg.inv(self.W) @ x


class mmICA(TypicalTransformer, BaseMMICA):
    base_algorithm = BaseMMICA

    def __init__(self, init_size=0, **kwargs):
        super().__init__(**kwargs)
        self.processing_queue = []
        self.init_size = init_size
        self.log = {'W': [], 't': []}

    def instance_get_params(self, deep=True):
        return dict(
            init_size=self.init_size,
            density_name=self.density_name,
            maxiter_cg=self.maxiter_cg,
            greedy=self.greedy,
            alpha=self.alpha,
            track_extra_info=self.track_extra_info,
            tol=self.tol
        )

    def pre_initialization_fit_for_X(self, X):
        if self.p is None:
            self.set_p(X.shape[1])
        else:
            self.partial_fit_for_X(X)

            if self.columns_seen > self.init_size:
                self.is_initialized = True

    def partial_fit_for_X(self, X):
        self.processing_queue.extend(X)
        if len(self.processing_queue) >= self.p:
            self.observe_new_batch(np.array(self.processing_queue).T)
            self.processing_queue = []

    def log_for_partial_fit(self, data, stream=0, pre_initialization=False):
        if not pre_initialization and self.input_streams[stream] == 'X' and self.log_level >= 1:
            self.log['W'].append(self.W.copy())
            self.log['t'].append(data.t)

    def transform_for_X(self, X):
        return self.unmix(X.T).T

    def inverse_transform_for_X(self, X):
        return self.remix(X.T).T
