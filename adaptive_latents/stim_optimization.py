from adaptive_latents.input_sources.autoregressor import AdamOptimizer
import jax
import numpy as np
import jax.numpy as jnp
from jax.nn import relu
import warnings

def loss(s, v, lam_1=1e-3):
    u = s
    loss = (
            - jnp.sqrt(jnp.linalg.norm(v.T @ s)) ** 2  # maximize dot product with the target vector
            + jnp.linalg.norm(s - v @ v.T @ s) ** 2  # minimize orthogonal component
            + jnp.linalg.norm(u, ord=1) * lam_1  # L1 penalty
    )
    ratio = (jnp.sqrt(jnp.linalg.norm(v.T @ s)) ** 2) / (jnp.linalg.norm(s - v @ v.T @ s) ** 2)
    return loss, ratio


class StimDesigner:
    def __init__(self, max_l0_norm=30, l0_norm_margin=5):
        self.grad_loss = None
        self.max_l0_norm = max_l0_norm
        self.convergence_threshold = 1e-3
        self.adam_learning_rate = 0.005
        self.starter_lam_1_guess = 10**-.5
        self.l0_norm_margin = l0_norm_margin
        self.log = []
        self._add_jited_functions()

    def _add_jited_functions(self):
        self.grad_loss = jax.jit(jax.value_and_grad(loss, has_aux=True))

    def design_stim(self, v, max_outer_iters=10, max_inner_iters=250, rng=None):
        assert len(v.shape) == 2
        assert self.max_l0_norm > 0
        if rng is None:
            rng = np.random.default_rng()


        lam_1_history = []
        loss_history = []
        s_history = []
        l0_history = []

        lam_1 = self.starter_lam_1_guess

        best_so_far = (None, -np.inf)

        for _ in range(max_outer_iters):
            loss_history.append([])
            s_history.append([])
            lam_1_history.append(lam_1)

            s = rng.uniform(size=(max(v.shape),)) * .1
            s_optimizer = AdamOptimizer(lr=self.adam_learning_rate)

            for i in range(max_inner_iters):
                (loss, aux), grad = self.grad_loss(s, v, lam_1=lam_1)
                s = s_optimizer.update(s,grad)
                s = relu(s)

                s_history[-1].append(s)
                loss_history[-1].append(loss)

                if np.isfinite(s).all() and  np.linalg.norm(s, ord=0) <= self.max_l0_norm and aux > best_so_far[1]:
                    best_so_far = (np.array(s), aux)

                if (~np.isfinite(s)).any() or (len(s_history[-1]) > 10 and jnp.linalg.norm(s_history[-1][-2] - s_history[-1][-1]) < self.convergence_threshold):
                    break

            l0 = np.linalg.norm(s,ord=0)
            l0_history.append(l0)
            if 0 <= self.max_l0_norm - l0 <= self.l0_norm_margin:
                break

            lam_1 = self.generate_next_lam_1(lam_1_history, l0_history)

        s = best_so_far[0]

        if s.max() > 0:
            s = np.array(s / s.max())

        self.log.append({'v':v, 's':s, 'loss_history':loss_history, 'lam_1_history':lam_1_history, 'l0_history':l0_history, 's_history':s_history})

        return s

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
