import numpy as np
from scipy.linalg import block_diag
from .transformer import TypicalTransformer
from adaptive_latents.regressions import BaseVanillaOnlineRegressor
from .utils import principle_angles, align_column_spaces
import warnings
import tqdm
from scipy.stats import special_ortho_group


class BaseSJPCA:
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
        # assert input_d % 2 == 0
        self.input_d = input_d
        self.H = self.make_H(self.input_d)
        self.reg = BaseVanillaOnlineRegressor(add_intercept=False)

        self.last_x = x

    def observe(self, x):
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
                # TODO: also permute planes?
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
        if order == 'C':
            X_tilde = np.zeros(shape=(m * n, n * n))
            for i in range(m):
                for j in range(n):
                    X_tilde[i * n + j, j * n:(j + 1) * n] = X[i]
        elif order == 'F':
            X_tilde = block_diag(*[X] * n)
        else:
            raise Exception("Input must be 'C' or 'F'")

        return X_tilde

    def project(self, x, project_up=False):
        U = self.get_U()
        return x @ (U if not project_up else U.T)




class sjPCA(TypicalTransformer, BaseSJPCA):
    base_algorithm = BaseSJPCA
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log = {'U': [], 't':[]}

    def instance_get_params(self, deep=True):
        return {}

    def pre_initialization_fit_for_X(self, X):
        if self.last_x is None: #  round 1
            self.initialize(X)
        else:
            self.observe(X)
            self.is_initialized = True


    def partial_fit_for_X(self, X):
        self.observe(X)

    def transform_for_X(self, X):
        return self.project(X)

    def inverse_transform_for_X(self, X):
        return self.project(X, project_up=True)

    def log_for_partial_fit(self, data, stream=0, pre_initialization=False):
        if not pre_initialization and self.input_streams[stream] == 'X' and self.log_level >= 1:
            self.log['U'].append(self.get_U())
            self.log['t'].append(data.t)

    def get_distance_from_subspace_over_time(self, subspace):
        assert self.log_level >= 1
        n = self.log['U'][0].shape[0]
        m = len(self.log['U'])
        distances = np.empty((m, n//2))
        for j, U in enumerate(self.log['U']):
            if U is None or np.any(np.isnan(U)):
                distances[j,:] = np.nan
                continue
            for plane_idx in range(n//2):
                sub_U = U[:, plane_idx*2: (plane_idx + 1)*2]
                distances[j, plane_idx] = np.abs(principle_angles(sub_U, subspace)).sum()
        # todo: divide by pi to normalize to 1?
        return distances, np.array(self.log['t'])

    def get_U_stability(self):
        assert self.log_level > 0
        Us = np.array(self.log['U'])
        t = np.array(self.log['t'])

        assert len(Us)
        dU = np.linalg.norm(np.diff(Us, axis=0), axis=1)[:,::2]
        return dU, t[1:]

    def plot_U_stability(self, ax):
        """
        Parameters
        ----------
        ax: matplotlib.axes.Axes
            the axes on which to plot the history
        """
        dU, t = self.get_U_stability()
        ax.plot(t, dU)
        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\Vert dU_{2i}\Vert$')
        ax.set_title(f"Numerical change in the bases of the planes of sjPCA")



def X_and_X_dot_from_data(X_all):
    """note: this is technically off-by-one for the way I normally think about it, but it's causal"""
    # todo: is this necessarily off-by-one?
    X_dot = np.diff(X_all, axis=0)
    X = X_all[1:]
    return X, X_dot


def generate_circle_embedded_in_high_d(rng, m=1000, n=4, stddev=1, shape=(10, 10)):
    t = np.linspace(0, (m / 10) * np.pi * 2, m + 1)
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
