import time
import functools

from adaptive_latents.input_sources.autoregressor import AdamOptimizer
import jax
import numpy as np
import jax.numpy as jnp
from jax.nn import relu
import warnings
from collections import deque

def loss(u, v, u_to_s_function, N=10, lam_1=1e-3):
    s = u_to_s_function(u)
    loss = (
            + 2*s.T@v@v.T@s
            - s.T@s
            #+ jnp.linalg.norm(v.T @ s) ** 2  # maximize dot product with the target vector
            #- jnp.linalg.norm(s - v @ v.T @ s) ** 2  # minimize orthogonal component
            - jnp.abs(jnp.linalg.norm(u, ord=1) - N)* lam_1  # L1 penalty
    )
    # ratio = (jnp.sqrt(jnp.linalg.norm(v.T @ s)) ** 2) / (jnp.linalg.norm(s - v @ v.T @ s) ** 2)
    return -loss #, ratio


class StimDesigner:
    def __init__(
            self,
            max_l0_norm=30,
            l0_norm_margin=5,
            adaptive_starter_lam_1=True,
            max_outer_loop_time_ms=10,
            max_inner_iters=500,
            rng_seed=0,
            should_log=False,
            convergence_threshold=0.01, #10**-.944,
            adam_learning_rate=0.1, #10**-.889,
            starter_lam_1_guess=0.01, #10**-.5,
    ):
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
        self.grad_loss = None
        self.max_l0_norm = max_l0_norm
        self.convergence_threshold = convergence_threshold
        self.adam_learning_rate = adam_learning_rate
        self.starter_lam_1_guess = starter_lam_1_guess
        self.max_outer_loop_time_ms = max_outer_loop_time_ms
        self.max_inner_iters = max_inner_iters
        self.adaptive_starter_lam_1 = adaptive_starter_lam_1
        self.l0_norm_margin = l0_norm_margin
        self.should_log = should_log
        self.log = []
        self._add_jited_functions()

    def _add_jited_functions(self):
        self.grad_loss = jax.jit(jax.value_and_grad(loss, has_aux=False))

    def design_stim(self, v, u_dimension, u_to_s_function=None):
        start_time = time.time()
        assert len(v.shape) == 2
        assert self.max_l0_norm > 0

        if u_to_s_function is None:
            grad_loss = self.grad_loss
        else:
            inner_loss = functools.partial(loss, u_to_s_function=u_to_s_function)
            grad_loss = jax.jit(jax.value_and_grad(inner_loss, has_aux=False))


        lam_1_history = []
        loss_history = []
        s_history = []
        l0_history = []

        lam_1 = self.starter_lam_1_guess

        best_so_far = (np.zeros(shape=(u_dimension,)), dict(lam_1=lam_1))

        while (time.time() - start_time) * 1000 < self.max_outer_loop_time_ms:
            loss_history.append([])
            s_history.append([])
            lam_1_history.append(lam_1)

            u = self.rng.uniform(size=(u_dimension,)) * .1
            s_optimizer = AdamOptimizer(lr=self.adam_learning_rate)
            s_history[-1].append(u)
            (loss_value), grad = grad_loss(u, v, N=self.max_l0_norm, lam_1=lam_1)
            loss_history[-1].append(loss_value)

            for i in range(self.max_inner_iters):
                (loss_value), grad = grad_loss(u, v, N=self.max_l0_norm, lam_1=lam_1)
                u = s_optimizer.update(u,grad)
                u = relu(u)

                s_history[-1].append(u)
                loss_history[-1].append(loss_value)

                if np.isfinite(u).all() and  np.linalg.norm(u, ord=0) <= self.max_l0_norm: #and aux > best_so_far[1]:
                    best_so_far = (np.array(u), dict(lam_1=lam_1))

                if (~np.isfinite(u)).any() or (len(s_history[-1]) > 20 and jnp.linalg.norm(s_history[-1][-2] - s_history[-1][-1]) < self.convergence_threshold) or np.abs(u).max() > 50:
                    break

            l0 = np.linalg.norm(u,ord=0)
            l0_history.append(l0)
            if 0 <= self.max_l0_norm - l0 <= self.l0_norm_margin:
                break

            lam_1 = self.generate_next_lam_1(lam_1_history, l0_history)

        u = best_so_far[0]

        if u.max() > 0:
            u = np.array(u / u.max())


        if self.adaptive_starter_lam_1:
            self.starter_lam_1_guess = best_so_far[1]['lam_1']

        if self.should_log:
            self.log.append({'time': time.time() - start_time, 'v':v, 'u':u, 'u_to_s_function_is_none':u_to_s_function is None, 'loss_history':loss_history, 'lam_1_history':lam_1_history, 'l0_history':l0_history, 's_history':s_history, 'predicted_s':u_to_s_function(u)})
        return u

    def generate_next_lam_1(self, lam_1_history, l0_history):
        # return np.random.default_rng().choice(np.logspace(-1, 0))
        l1h = np.array(lam_1_history)
        l0h = np.array(l0_history)

        too_lenient = l1h[l0h > self.max_l0_norm]
        too_strict = l1h[l0h < self.max_l0_norm]

        if len(too_lenient) == 0:
            return l1h.min() / 2
        if len(too_strict) == 0:
            return l1h.max() * 2

        # return geometric uniform between (too_lenient.max(), too_strict.min())
        return np.exp(np.mean(np.log((too_lenient.max(), too_strict.min()))))

    def __getstate__(self):
        return _unjax_state(self)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._add_jited_functions()

def _unjax_state(self):
    to_save = {}
    _pickle_changes = []
    for key, value in self.__dict__.items():
        if callable(value) and "jit" in str(value):
            _pickle_changes.append((key, "callable"))
            continue
        else:
            to_save[key] = value

    to_save["_pickle_changes"] = _pickle_changes
    return to_save
