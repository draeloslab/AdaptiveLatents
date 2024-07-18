import numpy as np
from scipy.linalg import rq


class proSVD:
    def __init__(self, k, decay_alpha=1, centering=False, mode='pro'):
        self.k = k
        self.decay_alpha = decay_alpha
        self.mode = mode
        self.centering = centering
        self.n_columns_seen = 0
        self.input_center = None

        self.Q = None
        self.B = None

    def initialize(self, A_init, Q_init=None, B_init=None):
        n, l1 = A_init.shape

        assert l1 >= self.k, "please init with # of cols >= k"
        # TODO: is this necessary?

        if self.centering:
            self.input_center = np.mean(A_init, axis=1)[:, None]
            A_init = A_init - self.input_center

        Q, B = np.linalg.qr(A_init, mode='reduced')

        self.Q = Q[:, :self.k] if Q_init is None else Q_init
        self.B = B[:self.k, :l1] if B_init is None else B_init

        self.n_columns_seen += A_init.shape[1]

    def run_on(self, X, initialization_size=None):
        initialization_size = initialization_size or X.shape[0]
        if self.Q is None:
            self.initialize(X[:, :initialization_size])
            X = X[:, initialization_size:]

        X_reduced = np.zeros((self.k, X.shape[1]))
        for i in np.arange(X.shape[1]):
            X_reduced[:, i:i + 1] = self.update_and_project(X[:, i:i + 1])

        return X_reduced

    def project(self, x):
        if self.centering:
            x = x - self.input_center
        return self.Q.T @ x

    def update_and_project(self, x):
        self.updateSVD(x)
        return self.project(x)

    def add_new_input_channels(self, n):
        self.Q = np.vstack([self.Q, np.zeros(shape=(n, self.Q.shape[1]))])

    def updateSVD(self, A, ref_basis=None):
        if self.centering:
            self.input_center = self.input_center + (A - self.input_center).mean(axis=1)[:, None] / (self.n_columns_seen + A.shape[1])
            A = A - self.input_center

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

        if self.mode == 'pro':  # Orthogonal Procrustes solution
            if ref_basis is not None:  # solution for a 'reference' basis
                Mu = ref_basis.T @ Q_hat @ U[:, :self.k]
            else:  # solution for minimum change from previous basis
                Mu = U[:self.k, :self.k]  # same as Mu = self.Q.T @ Q_hat @ U[:, :self.k]

            U_tilda, _, V_tilda_T = np.linalg.svd(Mu, full_matrices=False)
            Tu = U_tilda @ V_tilda_T
            Gu_1 = U[:, :self.k] @ Tu.T
        else:
            # Gu_1, Tu = rq(U[:, :self.k])  # Baker et al.
            Gu_1 = U[:, :self.k]  # Sequential KL
            Tu = Gu_1.T @ U[:, :self.k]

        Gv_1, Tv = rq(V[:, :self.k])

        self.Q = Q_hat @ Gu_1
        self.B = Tu @ np.diag(diag[:self.k]) @ Tv.T

        self.n_columns_seen += A.shape[1]
