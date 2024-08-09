import numpy as np
from scipy.linalg import rq
from .transformer import TypicalTransformer
from .utils import save_to_cache


class BaseProSVD:
    # todo: make this row-major
    def __init__(self, k=10, decay_alpha=1, whiten=False):
        self.k = k
        self.decay_alpha = decay_alpha
        self.whiten = whiten

        self.Q = None
        self.B = None
        self.n_samples_observed = 0

    def initialize(self, A_init):
        n, l1 = A_init.shape

        assert l1 >= self.k, "please init with # of cols >= k"
        assert n >= self.k, "k size doesn't make sense"

        Q, B = np.linalg.qr(A_init, mode='reduced')

        self.Q = Q[:, :self.k]
        self.B = B[:self.k, :self.k]
        self.n_samples_observed = l1

    def add_new_input_channels(self, n):
        self.Q = np.vstack([self.Q, np.zeros(shape=(n, self.Q.shape[1]))])

    def updateSVD(self, A):
        C = self.Q.T @ A
        A_perp = A - self.Q @ C
        Q_perp, B_perp = np.linalg.qr(A_perp, mode='reduced')

        Q_hat = np.concatenate((self.Q, Q_perp), axis=1)

        B_prev = np.concatenate((self.B, C), axis=1)
        tmp = np.zeros((B_perp.shape[0], self.B.shape[1]))
        tmp = np.concatenate((tmp, B_perp), axis=1)
        B_hat = np.concatenate((B_prev, tmp), axis=0)

        U, diag, V = np.linalg.svd(B_hat, full_matrices=False)

        diag *= self.decay_alpha

        Mu = U[:self.k, :self.k]  # same as Mu = self.Q.T @ Q_hat @ U[:, :self.k]

        U_tilda, _, V_tilda_T = np.linalg.svd(Mu, full_matrices=False)
        Tu = U_tilda @ V_tilda_T
        Gu_1 = U[:, :self.k] @ Tu.T

        Gv_1, Tv = rq(V[:, :self.k])

        self.Q = Q_hat @ Gu_1
        self.B = Tu @ np.diag(diag[:self.k]) @ Tv.T
        self.n_samples_observed += A.shape[1]

    def project(self, A):
        ret = self.Q.T @ A
        if self.whiten:
            # todo: this can be sped up with lapack.dtrtri or linalg.solve
            B = self.B/np.sqrt(self.n_samples_observed)
            ret = np.linalg.inv(B) @ ret
        return ret

    def get_cov_matrix(self):
        B = self.B/np.sqrt(self.n_samples_observed)
        return B @ B.T




class proSVD(TypicalTransformer, BaseProSVD):
    # todo: see if this use of wraps is a good idea
    # @wraps(TypicalTransformer.__init__)
    def __init__(self, init_size=None, **kwargs):
        super().__init__(**kwargs)
        self.init_size = init_size or self.k
        self.init_samples = []
        self.is_partially_initialized = False
        self.log = {'Q': [], 't': []}

    def pre_initialization_fit_for_X(self, X):
        if not self.is_partially_initialized:
            self.init_samples += list(X)
            if len(self.init_samples) >= self.init_size:
                self.initialize(np.array(self.init_samples).T)
                self.is_partially_initialized = True
        else:
            self.updateSVD(X.T)
        if self.is_partially_initialized and (not self.whiten or np.linalg.matrix_rank(self.B) == self.B.shape[0]):
            self.is_initialized = True


    def transform_for_X(self, X):
        return self.project(X.T).T

    def partial_fit_for_X(self, X):
        self.updateSVD(X.T)

    def log_for_partial_fit(self, data, stream=0, pre_initialization=False):
        if not pre_initialization:
            if self.log_level > 0:
                self.log['Q'].append(self.Q)
                self.log['t'].append(data.t) # todo: make this work again

    def get_Q_stability(self):
        assert self.log_level > 0
        Qs = np.array(self.log['Q'])

        t = np.arange(Qs.shape[0])
        if 't' in self.log:
            t = np.array(self.log['t'])

        assert len(Qs)
        dQ = np.linalg.norm(np.diff(Qs, axis=0), axis=1)
        return dQ, t[1:]

    def plot_Q_stability(self, ax):
        """
        Parameters
        ----------
        ax: matplotlib.axes.Axes
            the axes on which to plot the history
        """
        dQ, t = self.get_Q_stability()
        ax.plot(t, dQ)
        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\Vert dQ_i\Vert$')
        ax.set_title(f'Change in the columns of proSVD Q over time ({self.Q.shape[0]} -> {self.Q.shape[1]})')

    @save_to_cache("prosvd_data")
    @classmethod
    def offline_run_on_and_cache(cls, input_arr, **kwargs):
        pro = cls(**kwargs)
        return pro.offline_run_on(input_arr, convinient_return=True)


class RandomProjection(TypicalTransformer):
    def __init__(self, k=100, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.input_d = None

    def pre_initialization_fit_for_X(self, X):
        self.input_d = X.shape[1]
        # TODO: how to deal with randomness
        rng = np.random.default_rng()

        # TODO: other modes?
        self.U = rng.normal(size=(self.input_d, self.k), scale=1/(self.input_d * self.k))
        self.is_initialized = True

    def partial_fit_for_X(self, X):
        return

    def transform_for_X(self, X):
        return X @ self.U