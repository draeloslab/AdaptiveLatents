import warnings

import jax
import numpy as np
from jax import numpy as jnp


class AdamOptimizer:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = None

    def update(self, param, grad):
        if self.m is None:
            self.m = 0 * param
            self.v = 0 * param
            self.t = 0
        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        param = param - self.lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
        return param


class AR_K:
    # inspired by https://doi.org/10.48550/arXiv.2412.02529
    def __init__(self, *, k=1, rank_limit=None, rng=None, init_method='full_rank', iter_limit=500):
        # type: (AR_K, int, int | None, Any, Literal['full_rank', 'random'], int) -> None
        self.k = k
        self.rank_limit = rank_limit
        self.init_method = init_method
        self.iter_limit = iter_limit

        self.neuron_d = None
        self.stim_d = None

        self.As = None
        self.Bs = None
        self.v = None

        self._X = None
        self._Y = None

        self.rng = rng or np.random.default_rng(0)

    def fit(self, activity, stims):
        self.neuron_d = activity.shape[1]
        self.stim_d = stims.shape[1]

        assert self.rank_limit is None or not self.rank_limit > self.neuron_d

        Y = []
        X = []
        for i in range(self.k, activity.shape[0]):
            Y.append(activity[i])
            x = [activity[i - self.k:i].reshape(1, -1), stims[i - self.k:i].reshape(1, -1), [[1]]]
            X.append(np.hstack(x)[0])

        Y = np.array(Y)
        X = np.array(X)

        if self.rank_limit is None:
            As, Bs, v = self._fit_full_rank(X, Y)
        else:
            As, Bs, v = self._fit_reduced_rank(X, Y)

        self.As = As
        self.Bs = Bs
        self.v = v
        self._X = X
        self._Y = Y

    def _fit_full_rank(self, X, Y):
        beta, residuals, rank, s = np.linalg.lstsq(X, Y)

        As = beta[:self.neuron_d * self.k].reshape(self.k, self.neuron_d, self.neuron_d)
        Bs = beta[self.neuron_d * self.k:self.neuron_d * self.k + self.stim_d * self.k].reshape(self.k, self.stim_d,
                                                                                                self.neuron_d)
        v = beta[self.neuron_d * self.k + self.stim_d * self.k:]

        return As, Bs, v

    def make_initial_params_guess(self, X, Y):
        if self.init_method == 'full_rank':
            As, Bs, v = self._fit_full_rank(X, Y)

            B_U_s, B_S_s, B_Vh_s = np.linalg.svd(Bs, full_matrices=False)
            B_U_s = B_U_s * np.sqrt(B_S_s[..., None, :])
            B_Us = B_U_s[..., :self.rank_limit]
            B_Vh_s = np.sqrt(B_S_s[..., None]) * B_Vh_s
            B_V_s = B_Vh_s[..., :self.rank_limit, :].transpose(0, 2, 1)

            A_Ds = np.linalg.diagonal(As)
            A_nodiag_s = As - np.apply_along_axis(np.diag, -1, A_Ds)
            A_U_s, A_S_s, A_Vh_s = np.linalg.svd(A_nodiag_s, full_matrices=False)
            A_U_s = A_U_s * np.sqrt(A_S_s[..., None, :])
            A_U_s = A_U_s[..., :self.rank_limit]
            A_Vh_s = np.sqrt(A_S_s[..., None]) * A_Vh_s
            A_V_s = A_Vh_s[..., :self.rank_limit, :].transpose(0, 2, 1)

            # assert np.allclose(Bs, B_Us @ B_V_s.T)
            # assert np.allclose(As, A_U_s @ A_V_s.T + np.apply_along_axis(np.diag, -1, A_Ds))

            params = {
                'A_Ds': A_Ds,
                'A_Us': A_U_s,
                'A_Vs': A_V_s,
                'B_Us': B_Us,
                'B_Vs': B_V_s,
                'v': v
            }

        elif self.init_method == 'random':
            warnings.warn('this has been very unreliable in the past, check for stability')
            params = {
                'A_Ds': self.rng.normal(size=(self.k, self.neuron_d)),
                'A_Us': self.rng.normal(size=(self.k, self.neuron_d, self.rank_limit)),
                'A_Vs': self.rng.normal(size=(self.k, self.neuron_d, self.rank_limit)),

                'B_Us': self.rng.normal(size=(self.k, self.stim_d, self.rank_limit)),
                'B_Vs': self.rng.normal(size=(self.k, self.neuron_d, self.rank_limit)),
                'v': self.rng.normal(size=(1, self.neuron_d))
            }
            params = {k: v * 0.001 for k, v in params.items()}
        else:
            raise ValueError()

        return params

    def _fit_reduced_rank(self, X, Y):
        params = self.make_initial_params_guess(X, Y)

        def loss(params):
            As = jnp.apply_along_axis(jnp.diag, axis=1, arr=params['A_Ds']) + params['A_Us'] @ params['A_Vs'].transpose(
                0, 2, 1)
            Bs = params['B_Us'] @ params['B_Vs'].transpose(0, 2, 1)

            beta = jnp.vstack([As.reshape(-1, As.shape[-1]), Bs.reshape(-1, As.shape[-1]), params['v']])
            return jnp.linalg.norm(Y - X @ beta)

        optimizers = {k: AdamOptimizer() for k in params}

        grad_loss = jax.jit(jax.grad(loss))

        for i in range(self.iter_limit):
            grads = grad_loss(params)
            for k in params:
                params[k] = optimizers[k].update(params[k], grads[k])

        As = jnp.apply_along_axis(jnp.diag, axis=1, arr=params['A_Ds']) + params['A_Us'] @ params['A_Vs'].transpose(0, 2, 1)
        Bs = params['B_Us'] @ params['B_Vs'].transpose(0, 2, 1)
        v = params['v']
        return np.array(As), np.array(Bs), np.array(v)

    def predict(self, initial_state, stims=None, n_steps=100):
        if stims is None:
            stims = np.zeros((n_steps + self.k, self.Bs.shape[1]))
        assert initial_state.shape[0] >= self.k
        assert stims.shape[0] == n_steps + self.k

        new = np.zeros(shape=(n_steps + self.k, self.neuron_d)) * np.nan
        new[:self.k] = initial_state[-self.k:]

        for i in np.arange(n_steps) + self.k:
            new[i] = self.v
            new[i] = new[i] + ((new[i - self.k:i, None]) @ self.As).sum(axis=0)
            new[i] = new[i] + (stims[i - self.k:i, None] @ self.Bs).sum(axis=0)
        return new[-n_steps:]