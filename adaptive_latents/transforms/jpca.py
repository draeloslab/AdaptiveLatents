import numpy as np
from scipy.linalg import block_diag
from adaptive_latents.transforms import TransformerMixin
from adaptive_latents.regressions import VanillaOnlineRegressor
from .utils import save_to_cache, prosvd_data, align_column_spaces
import tqdm
from scipy.stats import special_ortho_group


class sjPCA:
    "a streaming implementation of jPCA"
    def __init__(self):
        self.is_initialized = False
        self.input_d = None
        self.H = None
        self.reg = None
        self.last_x = None
        self.last_U = None

    def initialize(self, x):
        input_d = x.shape[1]
        assert input_d % 2 == 0
        self.input_d = input_d
        self.H = self.make_H(self.input_d)
        self.reg = VanillaOnlineRegressor(input_d=self.H.shape[1], output_d=1, add_intercept=False)
        self.is_initialized = True

        self.last_x = x

    def observe(self, x):
        assert x.shape[0] == 1
        dx = x - self.last_x
        x_tilde = self.make_X_tilde(x)
        rows = x_tilde @ self.H
        for j in range(self.input_d):
            xx = rows[j]
            y = dx[0,j]
            self.reg.observe(xx, y)

        self.last_x = x

    def get_U(self):
        beta = self.reg.get_beta()
        n = self.input_d
        if np.any(np.isnan(beta)):
            return np.zeros((n, n)) * np.nan
        sksym = (self.H @ beta.ravel()).reshape(n, n)
        evals, evecs = np.linalg.eig(sksym)
        idx = np.argsort(np.abs(np.imag(evals)) + 1j * np.imag(evals))[::-1]
        evals, evecs = evals[idx], evecs[:, idx]

        U = np.zeros((n, n))
        for i in range(n // 2):
            v1 = evecs[:, i * 2]
            v2 = evecs[:, i*2 + 1]
            if np.sign(np.real(v1[0])) != np.sign(np.real(v2[0])):
                v2 = -v2
            # assert np.allclose(np.real(v1), np.real(v2))
            u1 = v1 + v2
            u2 = 1j * (v1-v2)
            u1 /= np.linalg.norm(u1)
            u2 /= np.linalg.norm(u2)
            # assert np.allclose(np.imag(u1),0)
            # assert np.allclose(np.imag(u2),0)
            U[:, i * 2] = np.real(u1)
            U[:, i*2 + 1] = np.real(u2)
            if self.last_U is not None and np.all(~np.isnan(self.last_U)):
                U[:, (i * 2):(i*2 + 2)], _ = align_column_spaces(U[:, (i * 2):(i*2 + 2)], self.last_U[:, (i * 2):(i*2 + 2)])
        self.last_U = U
        return U

    @staticmethod
    def make_H(d):
        h = []
        for i in range(0, d):
            for j in range(0, i):
                a = np.zeros((d, d))
                a[i, j] = 1
                a[j, i] = -1
                h.append(a.flatten())
        return np.column_stack(h)

    @staticmethod
    def make_X_tilde(X, order='C'):
        m, n = X.shape
        match order:
            case 'C':
                X_tilde = np.zeros(shape=(m * n, n * n))
                for i in range(m):
                    for j in range(n):
                        X_tilde[i*n + j, j * n:(j+1) * n] = X[i]
            case 'F':
                X_tilde = block_diag(*[X] * n)
            case _:
                raise Exception("Input must be 'C' or 'F'")

        return X_tilde

    def project(self, x):
        U = self.get_U()
        return x @ U


class TransformerSJPCA(TransformerMixin, sjPCA):
    def partial_fit_transform(self, data, stream=0):
        if self.input_streams[stream] == 'X':
            if not self.is_initialized:
                if not np.any(np.isnan(data)):
                    self.initialize(data)
                return np.nan * data
            else:
                self.observe(data)
                return self.project(data)
        else:
            return data

    def transform(self, data, stream=0):
        if self.input_streams[stream] == 'X':
            if not self.is_initialized:
                return np.nan * data
            else:
                return self.project(data)
        else:
            return data


def X_and_X_dot_from_data(X_all):
    """note: this is technically off-by-one for the way I normally think about it, but it's causal"""
    # todo: is this necessarily off-by-one?
    X_dot = np.diff(X_all, axis=0)
    X = X_all[1:]
    return X, X_dot


def generate_circle_embedded_in_high_d(rng, m=1000, n=4, stddev=1, shape=(10, 10)):
    t = np.linspace(0, m / 50 * np.pi * 2, m + 1)
    circle = np.column_stack([np.cos(t), np.sin(t)]) @ np.diag(shape)
    C = special_ortho_group(dim=n, seed=rng).rvs()[:, :2]
    X_all = (circle @ C.T) + rng.normal(size=(m + 1, n)) * stddev
    X, X_dot = X_and_X_dot_from_data(X_all)
    return X, X_dot, dict(C=C)


# def generate_by_circle(rng, m=1000, n=4):
#     t = np.linspace(0, m/50*np.pi*2, m+1)
#     circle = np.column_stack([np.cos(t),np.sin(t)]) @ np.diag([20,10])
#     C = special_ortho_group(dim=n,seed=rng).rvs()[:,:2]
#     X_all = (circle @ C.T) + rng.normal(size=(m+1,n))
#     X, X_dot = from_data(X_all)
#     return X, X_dot, dict(C=C)
