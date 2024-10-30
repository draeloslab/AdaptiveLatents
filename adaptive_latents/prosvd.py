import numpy as np
from scipy.linalg import rq
from .transformer import TypicalTransformer
from .utils import save_to_cache, principle_angles


class BaseProSVD:
    # todo: make this row-major
    def __init__(self, k=1, decay_alpha=1, whiten=False):
        self.k = k
        self.decay_alpha = decay_alpha
        self.whiten = whiten

        self.Q = None
        self.R = None
        self.n_samples_observed = 0

    def initialize(self, x):
        sample_d, n_samples = x.shape

        assert n_samples >= self.k, "please init with # of cols >= k"
        assert sample_d >= self.k, "k size doesn't make sense"

        Q, R = np.linalg.qr(x, mode='reduced')

        self.Q = Q[:, :self.k]  # TODO: why isn't Q square here?
        self.R = R[:self.k, :self.k]
        self.n_samples_observed = n_samples

    def add_new_input_channels(self, n):
        self.Q = np.vstack([self.Q, np.zeros(shape=(n, self.Q.shape[1]))])

    def updateSVD(self, x):
        x_along = self.Q.T @ x
        x_orth = x - self.Q @ x_along
        x_orth_q, x_orth_r = np.linalg.qr(x_orth, mode='reduced')

        q_new = np.hstack([self.Q, x_orth_q])
        r_new = np.block([
            [self.R,                                         x_along],
            [np.zeros((x_orth_r.shape[0], self.R.shape[1])), x_orth_r]
        ])

        u_high_d, diag_high_d, vh_high_d = np.linalg.svd(r_new, full_matrices=False)

        u_low_d = u_high_d[:,:self.k]
        vh_low_d = vh_high_d[:,:self.k]
        diag_low_d = diag_high_d[:self.k]

        diag_low_d *= self.decay_alpha

        # if 'alignment_method' == 'procrustean':

        # The new basis is `q_new @ u_low_d`; to align it to `X` we would do the SVD of `X.T @ (q_new @ u_low_d)`.
        # Since we want to align to `self.Q`, we would usually use `self.Q.T @ q_new @ u_low_d`, but we can simplify
        # because (self.Q.T @ q_new) has a lot of cancellations (see their definitions).
        temp = np.linalg.svd(u_low_d[:self.k, :], full_matrices=False)
        u_stabilizing_rotation = temp[0] @ temp[2]
        u_low_d_stabilized = u_low_d @ u_stabilizing_rotation.T

        # TODO: we don't actually stabilize anything here, I think this can be dropped
        vh_low_d_stabilized, vh_stabilizing_rotation = rq(vh_low_d)

        # elif 'alignment_method' == 'Baker 2012':
        #     # Baker refers to e.g. https://doi.org/10.1016/j.laa.2011.07.018
        #     u_low_d_stabilized, u_stabilizing_rotation = rq(u_low_d)
        #     vh_low_d_stabilized, vh_stabilizing_rotation = rq(vh_low_d)
        # elif 'alignment_method' == 'sequential KLT':
        #     # KLT is in the original proSVD code, not sure what the source is
        #     u_low_d_stabilized = u_low_d
        #     u_stabilizing_rotation = u_low_d.T @ u_low_d  # identity matrix
        #
        #     vh_low_d_stabilized = vh_low_d
        #     vh_stabilizing_rotation = vh_low_d.T @ vh_low_d

        self.Q = q_new @ u_low_d_stabilized
        self.R = (u_stabilizing_rotation * diag_low_d) @ vh_stabilizing_rotation.T

        self.n_samples_observed *= self.decay_alpha
        self.n_samples_observed += x.shape[1]

    def project_down(self, x):
        ret = self.Q.T @ x
        if self.whiten:
            # todo: this can be sped up with lapack.dtrtri or linalg.solve
            R = self.R / np.sqrt(self.n_samples_observed)
            ret = np.linalg.inv(R) @ ret
        return ret

    def project_up(self, x):
        if self.whiten:
            R = self.R / np.sqrt(self.n_samples_observed)
            x = R @ x
        return self.Q @ x

    def get_cov_matrix(self):
        R = self.R / np.sqrt(self.n_samples_observed)
        return R @ R.T



class proSVD(TypicalTransformer, BaseProSVD):
    base_algorithm = BaseProSVD

    def __init__(self, init_size=None, **kwargs):
        super().__init__(**kwargs)
        self.init_size = init_size or self.k
        self.on_nan_width = self.k
        self.init_samples = []
        self.is_partially_initialized = False
        self.log = {'Q': [], 't': []}


    def instance_get_params(self, deep=True):
        return dict(k=self.k, decay_alpha=self.decay_alpha, whiten=self.whiten, init_size=self.init_size)


    def pre_initialization_fit_for_X(self, X):
        if not self.is_partially_initialized:
            self.init_samples += list(X)
            if len(self.init_samples) >= self.init_size:
                self.initialize(np.array(self.init_samples).T)
                self.is_partially_initialized = True
        else:
            self.updateSVD(X.T)
        if self.is_partially_initialized and (not self.whiten or np.linalg.matrix_rank(self.R) == self.R.shape[0]):
            self.is_initialized = True


    def transform_for_X(self, X):
        return self.project_down(X.T).T

    def inverse_transform_for_X(self, X):
        return self.project_up(X.T).T

    def partial_fit_for_X(self, X):
        self.updateSVD(X.T)

    def log_for_partial_fit(self, data, stream=0):
        if self.is_initialized:
            if self.log_level > 0:
                self.log['Q'].append(self.Q)
                self.log['t'].append(data.t)

    def get_distance_from_subspace_over_time(self, subspace):
        assert self.log_level >= 1
        m = len(self.log['Q'])
        distances = np.empty(m)
        for j, Q in enumerate(self.log['Q']):
            if np.any(np.isnan(Q)):
                distances[j] = np.nan
                continue
            distances[j] = np.abs(principle_angles(Q, subspace)).sum()
        return distances, np.array(self.log['t'])

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

    def __init__(self, rng_seed=0, k=100, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.input_d = None
        self.rng_seed = rng_seed
        self.rng: np.random.Generator = np.random.default_rng(self.rng_seed)
        self.random_matrix = None
        self.inv = None

    def pre_initialization_fit_for_X(self, X):
        self.input_d = X.shape[1]

        # TODO: other modes?
        self.random_matrix = self.rng.normal(size=(self.input_d, self.k), scale=1 / (self.input_d * self.k))
        self.is_initialized = True

    def partial_fit_for_X(self, X):
        pass

    def instance_get_params(self, deep=True):
        return dict(k=self.k, rng_seed=self.rng_seed)

    def transform_for_X(self, X):
        return X @ self.random_matrix

    def inverse_transform_for_X(self, X):
        if self.inv is None:
            self.inv = np.linalg.pinv(self.random_matrix)
        return X @ self.inv