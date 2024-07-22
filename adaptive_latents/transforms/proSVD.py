import numpy as np
from scipy.linalg import rq
from .transformer import TransformerMixin


class proSVD:
    # todo: make this row-major
    def __init__(self, k=10, decay_alpha=1, whiten=False):
        self.k = k
        self.decay_alpha = decay_alpha
        self.whiten = whiten

        self.Q = None
        self.B = None
        self.n_samples_observed = 0
        self.is_initialized = False

    def initialize(self, A_init):
        n, l1 = A_init.shape

        assert l1 >= self.k, "please init with # of cols >= k"

        Q, B = np.linalg.qr(A_init, mode='reduced')

        self.Q = Q[:, :self.k]
        self.B = B[:self.k, :l1]
        self.n_samples_observed = l1
        self.is_initialized = True

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


class TransformerProSVD(TransformerMixin, proSVD):
    def __init__(self, k=10, decay_alpha=1, whiten=False, init_size=None, input_streams=None, output_streams=None, log_level=0):
        input_streams = input_streams or {0: 'X'}
        self.init_size = init_size or k
        self.init_samples = []
        super().__init__(k=k, whiten=whiten, decay_alpha=decay_alpha, input_streams=input_streams, output_streams=output_streams, log_level=log_level)

    def transform(self, data, stream=0):
        if self.input_streams[stream] == 'X':
            if not self.is_initialized:
                return np.nan * data
            return self.project(data)
        else:
            return data

    def partial_fit_transform(self, data, stream=0):
        if self.input_streams[stream] == 'X':
            if not self.is_initialized:
                self.init_samples += list(data)
                if len(self.init_samples) >= self.init_size:
                    self.initialize(np.array(self.init_samples).T)
                    # return self.transform(data.T, stream).T
                    # todo: error here
                return np.nan * data

            self.updateSVD(data.T)

            return self.transform(data.T, stream).T
        else:
            return data