import numpy
import jax.numpy as jnp
from collections import deque
from jax import jit, vmap
from jax import nn, random
import jax
import warnings
from matplotlib.patches import Ellipse
from math import atan2
from scipy.linalg import fractional_matrix_power
from .config import use_config_defaults
from .transformer import StreamingTransformer
from .timed_data_source import ArrayWithTime
import copy

# TODO: save frozen vs estimator frozen
# TODO: make this a parameter?
epsilon = 1e-10


class BaseBubblewrap:
    @use_config_defaults
    # note the defaults in this signature are overridden by the defaults in adaptive_latents_config
    def __init__(self, num=1000, seed=42, M=30, lam=1, nu=1e-2, eps=3e-2, B_thresh=1e-4, step=1e-6, n_thresh=5e-4, go_fast=False, copy_row_on_teleport=True, num_grad_q=1, sigma_orig_adjustment=0):

        self.N = num  # Number of nodes
        self.seed = seed
        self.lam_0 = lam
        self.nu = nu
        self.M = M

        self.eps = eps
        self.B_thresh = B_thresh
        self.n_thresh = n_thresh
        self.step = step
        self.copy_row_on_teleport = copy_row_on_teleport
        self.num_grad_q = num_grad_q
        self.sigma_orig_adjust = sigma_orig_adjustment

        self.go_fast = go_fast

        self.key = random.PRNGKey(self.seed)
        numpy.random.seed(self.seed)
        # TODO: change this to use the `rng` system

        # observations of the data; M is how many to keep in history
        self.obs = Observations(M=M, go_fast=go_fast)
        self._add_jited_functions()
        self.mu_orig = None

        self.is_initialized = False
        self.sfrozen = False

        self.backend_note = None
        self.precision_note = None
        if not self.go_fast:
            from jax.lib import xla_bridge
            self.backend_note = xla_bridge.get_backend().platform

            x = jax.random.uniform(jax.random.key(0), (1,), dtype=jnp.float64)
            self.precision_note = x.dtype
            if self.precision_note != jnp.float64:
                warnings.warn("You should probably run jax with 64-bit floats.")
                # raise FloatingPointError("You should probably run jax with 64-bit floats.")

    def init_nodes(self):
        ### Based on observed data so far of length M
        self.d = self.obs.saved_obs[-1].size
        self.mu = jnp.zeros((self.N, self.d))

        com = center_mass(self.mu)
        if len(self.obs.saved_obs) > 1:
            obs_com = center_mass(self.obs.saved_obs)
        else:
            ## this section for if we init nodes with no data
            obs_com = 0
            self.obs.curr = com
            self.obs.obs_com = com

        self.mu += obs_com

        prior = (1 / self.N) * jnp.ones(self.N)

        self.alpha = self.lam_0 * prior
        self.last_alpha = self.alpha.copy()
        self.lam = self.lam_0 * prior
        self.n_obs = 0 * self.alpha

        self.mu_orig = self.mu.copy()
        self.mus_orig = self.get_mus0(self.mu_orig)

        ### Initialize model parameters (A,En,...)
        self.A = jnp.ones((self.N, self.N)) - jnp.eye(self.N)
        self.A /= jnp.sum(self.A, axis=1)
        self.B = jnp.zeros((self.N))
        self.En = jnp.zeros((self.N, self.N))

        self.S1 = jnp.zeros((self.N, self.d))
        self.S2 = jnp.zeros((self.N, self.d, self.d))

        self.log_A = jnp.zeros((self.N, self.N))

        fullSigma = numpy.zeros((self.N, self.d, self.d), dtype="float32")
        self.L = numpy.zeros((self.N, self.d, self.d))
        self.L_diag = numpy.zeros((self.N, self.d))
        var = jnp.diag(jnp.var(jnp.array(self.obs.saved_obs), axis=0))
        for n in numpy.arange(self.N):
            fullSigma[n] = var * (self.nu + self.d + 1) / (self.N**(2 / self.d))

            ## Optimization is done with L split into L_lower and L_diag elements
            ## L is defined using cholesky of precision matrix, NOT covariance
            L = jnp.linalg.cholesky(fullSigma[n])
            self.L[n] = jnp.linalg.inv(L).T
            self.L_diag[n] = jnp.log(jnp.diag(self.L[n]))
        self.L_lower = jnp.tril(self.L, -1)
        self.sigma_orig = fullSigma[0]

        ## for adam gradients
        self.m_mu = jnp.zeros_like(self.mu)
        self.m_L_lower = jnp.zeros_like(self.L_lower)
        self.m_L_diag = jnp.zeros_like(self.L_diag)
        self.m_A = jnp.zeros_like(self.A)

        self.v_mu = jnp.zeros_like(self.mu)
        self.v_L_lower = jnp.zeros_like(self.L_lower)
        self.v_L_diag = jnp.zeros_like(self.L_diag)
        self.v_A = jnp.zeros_like(self.A)

        ## Variables for keeping track of dead nodes
        self.dead_nodes = jnp.arange(0, self.N).tolist()
        self.dead_nodes_ind = self.n_thresh * numpy.ones(self.N)
        self.current_node = 0

        self.t = 1  # todo: what is this doing in ADAM?
        self.is_initialized = True

    def _add_jited_functions(self):
        # doing this here allows us to add the jax-specific functions back after pickling
        self.get_mus0 = jit(vmap(get_mus, 0))

        ## Set up gradients
        self.grad_all = jit(vmap(jit(jax.value_and_grad(Q_j, argnums=(0, 1, 2, 3), has_aux=True)), in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, 0)))

        ## Other jitted functions
        self.logB_jax = jit(vmap(single_logB, in_axes=(None, 0, 0, 0)))
        self.expB_jax = jit(expB)
        self.update_internal_jax = jit(update_internal)
        self.kill_nodes = jit(kill_dead_nodes)
        self._pred_ahead = jit(pred_ahead, static_argnames=['steps_ahead'])
        self.sum_me = jit(sum_me)
        self.compute_L = jit(vmap(get_L, (0, 0)))
        self.get_amax = jit(amax)
        self._get_entropy = jit(entropy, static_argnames=['steps_ahead'])

    def observe(self, x):
        # Get new data point and update observation history

        ## Do all observations, and then update mu0, sigma0
        self.obs.new_obs(x)

        if not self.go_fast and self.obs.cov is not None and self.mu_orig is not None:
            lamr = 0.02  # this is $\lambda$ from the paper
            eta = jnp.sqrt(lamr * jnp.diag(self.obs.cov))  # this is $\nu$ from the paper

            self.mu_orig = (1 - lamr) * self.mu_orig + lamr * self.obs.mean + \
                            eta * numpy.random.normal(size=(self.N, self.d))
            self.sigma_orig = self.obs.cov * (self.nu + self.d + 1) / \
                            ((self.N + self.sigma_orig_adjust) ** (2 / self.d))

    def e_step(self):
        # take E step; after observation
        x = self.obs.curr
        self.beta = 1 + 10 / (self.t + 1)
        self.B = self.logB_jax(x, self.mu, self.L, self.L_diag)
        self.update_B(x)
        self.gamma, self.alpha, self.En, self.S1, self.S2, self.n_obs = self.update_internal_jax(self.A, self.B, self.alpha, self.En, self.eps, self.S1, x, self.S2, self.n_obs)

        if not self.go_fast and jnp.any(jnp.isnan(self.alpha)):
            # this sometimes happens when the input data has a singular covariance matrix
            raise Exception("There's a NaN in the alphas, something's wrong.")
        self.t += 1

    def update_B(self, x):
        if jnp.max(self.B) < self.B_thresh:
            if not (self.dead_nodes):
                target = numpy.argmin(self.n_obs)
                n_obs = numpy.array(self.n_obs)
                n_obs[target] = 0
                self.n_obs = n_obs
                self.remove_dead_nodes()
            node = self.teleport_node(x)
            self.B = self.logB_jax(x, self.mu, self.L, self.L_diag)
        self.current_node, self.B = self.expB_jax(self.B)

    def remove_dead_nodes(self):
        ma = (self.n_obs + self.dead_nodes_ind) < self.n_thresh

        if ma.any():
            ind2 = self.get_amax(ma)

            self.n_obs, self.S1, self.S2, self.En, self.log_A = self.kill_nodes(ind2, self.n_thresh, self.n_obs, self.S1, self.S2, self.En, self.log_A)
            actual_ind = int(ind2)
            self.dead_nodes.append(actual_ind)
            self.dead_nodes_ind[actual_ind] = self.n_thresh

    def teleport_node(self, x):
        node = self.dead_nodes.pop(0)

        mu = numpy.array(self.mu)
        mu[node] = x
        self.mu = mu

        alpha = numpy.array(self.alpha)
        alpha[node] = 1
        self.alpha = alpha

        self.dead_nodes_ind[node] = 0

        self.n_obs.at[node].set(1.0)

        if self.copy_row_on_teleport:
            # TODO: other updates here?
            nearest_bubble = numpy.argsort(numpy.linalg.norm(self.mu - x, axis=1))[1]
            A = numpy.array(self.A)
            A[node] = A[nearest_bubble]
            self.A = A

        return node

    def grad_Q(self):
        for _ in range(self.num_grad_q):
            div = 1 + self.sum_me(self.En)
            (self.Q, self.Q_parts), (self.grad_mu, self.grad_L_lower, self.grad_L_diag, self.grad_A) = \
            self.grad_all(self.mu, self.L_lower, self.L_diag, self.log_A, self.S1, self.lam, self.S2,
                          self.n_obs, self.En, self.nu, self.sigma_orig, self.beta, self.d, self.mu_orig)

            # this line is for debugging purposes; you can step through the inside of grad_all for a single bubble
            # _Q_j(self.mu[0], self.L_lower[0], self.L_diag[0], self.log_A[0], self.S1[0], self.lam[0], \
            # self.S2[0], self.n_obs[0], self.En[0], self.nu, self.sigma_orig, self.beta, self.d, self.mu_orig[0])

            self.run_adam(self.grad_mu / div, self.grad_L_lower / div, self.grad_L_diag / div, self.grad_A / div)

            self.A = sm(self.log_A)

            self.L = self.compute_L(self.L_diag, self.L_lower)

    def run_adam(self, mu, L, L_diag, A):
        ## inputs are gradients
        self.m_mu, self.v_mu, self.mu = single_adam(self.step, self.m_mu, self.v_mu, mu, self.t, self.mu)
        self.m_L_lower, self.v_L_lower, self.L_lower = single_adam(self.step, self.m_L_lower, self.v_L_lower, L, self.t, self.L_lower)
        self.m_L_diag, self.v_L_diag, self.L_diag = single_adam(self.step, self.m_L_diag, self.v_L_diag, L_diag, self.t, self.L_diag)
        self.m_A, self.v_A, self.log_A = single_adam(self.step, self.m_A, self.v_A, A, self.t, self.log_A)

    def unevaluated_log_pred_p(self, steps):
        if not self.is_initialized:
            return lambda x: numpy.nan

        assert round(steps) == steps

        mu = numpy.array(self.mu)
        L = numpy.array(self.L)
        L_diag = numpy.array(self.L_diag)
        A = numpy.array(self.A)
        alpha = numpy.array(self.alpha)

        def f(future_point):
            b = self.logB_jax(future_point, mu, L, L_diag)
            AT = jnp.linalg.matrix_power(A, steps)
            p = jnp.log(alpha @ AT @ jnp.exp(b) + 1e-16)
            return numpy.array(p)
        return f

    def log_pred_p(self, future_point, n_steps):
        if not self.is_initialized:
            return numpy.nan
        b = self.logB_jax(future_point, self.mu, self.L, self.L_diag)

        assert round(n_steps) == n_steps
        n_steps = int(n_steps)
        p = self._pred_ahead(b, self.A, self.alpha, n_steps)
        # AT = fractional_matrix_power(self.A, n_steps)
        # p = jnp.log(self.alpha @ AT @ jnp.exp(b) + 1e-16)
        return numpy.array(p)

    def entropy(self, n_steps, alpha=None):
        if not self.is_initialized:
            return numpy.nan
        if alpha is None:
            alpha = self.alpha

        assert round(n_steps) == n_steps
        n_steps = int(n_steps)
        e = self._get_entropy(self.A, alpha, n_steps)
        # AT = fractional_matrix_power(self.A, n_steps)
        # one = alpha @ AT
        # e = -jnp.sum(one.dot(jnp.log2(alpha @ AT)))

        return numpy.array(e)


    def sfreeze(self):
        self.sfrozen = True
        self.obs.sfreeze()

    def __getstate__(self):
        return _unjax_state(self)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not state["sfrozen"]:
            self._add_jited_functions()
            if not self.go_fast:
                from jax.lib import xla_bridge
                self.backend_note += " " + xla_bridge.get_backend().platform

    default_clock_parameters = dict(
        num=8,
        lam=1e-3,
        nu=1e-3,
        eps=1e-4,
        step=8e-2,
        M=100,
        B_thresh=-5,
        go_fast=False,
        seed=42,
        num_grad_q=1,
        copy_row_on_teleport=True,
        sigma_orig_adjustment=0,
        n_thresh=5e-4,
    )


beta1 = 0.99
beta2 = 0.999

### A ton of jitted functions for fast code execution


@jit
def single_adam(step, m, v, grad, t, val):
    m = beta1*m + (1-beta1) * grad
    v = beta2*v + (1-beta2) * grad**2
    m_hat = m / (1 - jnp.power(beta1, t + 1))
    v_hat = v / (1 - jnp.power(beta2, t + 1))
    update = step * m_hat / (jnp.sqrt(v_hat) + epsilon)
    val -= update
    return m, v, val


@jit
def sm(log_A):
    return nn.softmax(log_A, axis=1)


@jit
def sum_me(En):
    return jnp.sum(En)


@jit
def amax(A):
    return jnp.argmax(A)


@jit
def get_L(x, y):
    return jnp.tril(jnp.diag(jnp.exp(x) + epsilon) + jnp.tril(y, -1))


@jit
def get_mus(mu):
    return jnp.outer(mu, mu)


@jit
def get_ld(L):
    return -2 * jnp.sum(L)


def _Q_j(mu, L_lower, L_diag, log_A, S1, lam, S2, n_obs, En, nu, sigma_orig, beta, d, mu_orig):
    L = jnp.tril(jnp.diag(jnp.exp(L_diag) + epsilon) + jnp.tril(L_lower, -1))
    sig_inv = L @ L.T
    mus = jnp.outer(mu, mu)
    mus_orig = jnp.outer(mu_orig, mu_orig)
    ld = -2 * jnp.sum(L_diag)

    # todo: this list structure could be optimized
    to_sum = [None, None, None, None]
    to_sum[0] = ((S1 + lam*mu_orig).dot(sig_inv).dot(mu))
    to_sum[1] = ((-1 / 2) * jnp.trace(((sigma_orig + S2 + lam*mus_orig + (lam+n_obs) * mus) @  # NB: this is where a GPU numerical problem was cropping up at one point
                                       sig_inv)))
    to_sum[2] = ((-1 / 2) * (nu+n_obs+d+2) * ld)
    to_sum[3] = (jnp.sum((En+beta-1) * nn.log_softmax(log_A)))
    summed = to_sum[0] + to_sum[1] + to_sum[2] + to_sum[3]
    return -jnp.sum(summed), to_sum


Q_j = jit(_Q_j)


@jit
def single_logB(x, mu, L, L_diag):
    n = mu.shape[0]
    B = (-1 / 2) * jnp.linalg.norm((x-mu) @ L)**2 - (n/2) * jnp.log(2 * jnp.pi) + jnp.sum(L_diag)
    return B


# @jit
def expB(B):
    max_Bind = jnp.argmax(B)
    current_node = max_Bind
    B -= B[max_Bind]
    B = jnp.exp(B)
    # B = B.at[10:].set(0)
    return current_node, B


@jit
def update_internal(A, B, last_alpha, En, eps, S1, obs_curr, S2, n_obs):
    gamma = B * A / (last_alpha.dot(A).dot(B) + 1e-16)
    alpha = last_alpha.dot(gamma)
    En = gamma * last_alpha[:, jnp.newaxis] + (1-eps) * En
    S1 = (1-eps) * S1 + alpha[:, jnp.newaxis] * obs_curr
    S2 = (1-eps) * S2 + alpha[:, jnp.newaxis, jnp.newaxis] * (obs_curr[:, jnp.newaxis] * obs_curr.T)
    n_obs = (1-eps) * n_obs + alpha
    return gamma, alpha, En, S1, S2, n_obs


@jit
def kill_dead_nodes(ind2, n_thresh, n_obs, S1, S2, En, log_A):
    N = n_obs.shape[0]
    d = S1.shape[1]
    n_obs = n_obs.at[ind2].set(0)
    S1 = S1.at[ind2].set(jnp.zeros(d))
    S2 = S2.at[ind2].set(jnp.zeros((d, d)))
    log_A = log_A.at[ind2].set(jnp.zeros(N))
    log_A = log_A.at[:, ind2].set(jnp.zeros(N))
    return n_obs, S1, S2, En, log_A


# gets jit-ed later
def pred_ahead(B, A, alpha, steps_ahead):
    AT = jnp.linalg.matrix_power(A, steps_ahead)
    return jnp.log(alpha @ AT @ jnp.exp(B) + 1e-16)


# gets jit-ed later
def entropy(A, alpha, steps_ahead):
    AT = jnp.linalg.matrix_power(A, steps_ahead)
    one = alpha @ AT
    return -jnp.sum(one.dot(jnp.log2(alpha @ AT)))


def center_mass(points):
    return jnp.mean(jnp.array(points), axis=0)


class Observations:
    # TODO: get rid of this object?
    def __init__(self, M=5, go_fast=True):
        self.M = M  # how many observed points to hold in memory
        self.go_fast = go_fast

        self.curr = None
        self.saved_obs = deque(maxlen=self.M)

        self.mean = None
        self.last_mean = None

        self.cov = None

        self.n_obs = 0

        self.sfrozen = False

    def new_obs(self, coord_new):
        self.curr = coord_new
        self.saved_obs.append(self.curr)
        self.n_obs += 1

        if not self.go_fast:
            if self.mean is None:
                self.mean = self.curr.copy()
            else:
                self.last_mean = self.mean.copy()
                self.mean = update_mean(self.mean, self.curr, self.n_obs)

            if self.n_obs > 2:
                if self.cov is None:
                    self.cov = jnp.cov(jnp.array(self.saved_obs).T, bias=True)
                else:
                    self.cov = update_cov(self.cov, self.last_mean, self.curr, self.mean, self.n_obs)

    def sfreeze(self):
        self.sfrozen = True

    def __getstate__(self):
        return _unjax_state(self)


def _unjax_state(self):
    to_save = {}
    _pickle_changes = []
    for key, value in self.__dict__.items():
        if callable(value) and "jit" in str(value):
            _pickle_changes.append((key, "callable"))
            continue

        elif self.sfrozen and "jax" in str(type(value)) and "Array" in str(type(value)):
            to_save[key] = numpy.array(value)
            _pickle_changes.append((key, "unjaxed"))
        else:
            to_save[key] = value

    to_save["_pickle_changes"] = _pickle_changes
    return to_save


@jit
def update_mean(mean, curr, n_obs):
    return mean + (curr-mean) / n_obs


# @jit # TODO: profile this, and maybe bring it back
def update_cov(cov, last, curr, mean, n):
    lastm = get_mus(last)
    currm = get_mus(mean)
    curro = get_mus(curr)
    f = (n-1) / n
    return f * (cov+lastm) + (1-f) * curro - currm


class Bubblewrap(StreamingTransformer, BaseBubblewrap):
    base_algorithm = BaseBubblewrap

    def __init__(self, n_steps_to_predict=1, input_streams=None, **kwargs):
        input_streams = input_streams or {0 : 'X'}
        super().__init__(input_streams=input_streams, **kwargs)
        self.unevaluated_predictions = {}
        self.dt = None
        self.last_timepoint = None
        self.n_steps_to_predict = n_steps_to_predict

    def partial_fit_transform(self, data, stream=0, return_output_stream=False):
        original_data = None
        if self.log_level > 0:
            original_data = copy.deepcopy(data)
        ret = self._partial_fit_transform(data, stream, return_output_stream)
        self.log_for_partial_fit(original_data if original_data is not None else data, stream)
        return ret

    def _partial_fit_transform(self, data, stream=0, return_output_stream=False):
        if self.input_streams[stream] == 'X':
            if hasattr(data, 't') and self.last_timepoint is not None and self.dt is not None:
                new_dt = data.t - self.last_timepoint
                larger, smaller = sorted([self.dt, new_dt])
                assert larger / smaller < 1.1

            output = []
            for i in range(len(data)): # todo: how to deal with t's within a data matrix
                # partial fit
                datapoint = data[i]
                if not jnp.isnan(datapoint).any():
                    # main path (no nans)
                    self.observe(datapoint)

                    if not self.is_initialized and self.obs.n_obs > self.M:
                        self.init_nodes()

                    if self.is_initialized:
                        self.e_step()
                        self.grad_Q()
                elif self.is_initialized:
                    # if there is a nan
                    self.alpha = self.alpha @ self.A

                # transform
                if self.is_initialized:
                    o = numpy.array(self.alpha)
                else:
                    o = numpy.zeros(self.N) * numpy.nan
                output.append(o)

            if isinstance(data, ArrayWithTime):
                data = ArrayWithTime(output, data.t)

            if hasattr(data, 't'):
                if self.last_timepoint is not None:
                    self.dt = data.t - self.last_timepoint
                self.last_timepoint = data.t

        stream = self.output_streams[stream]
        if return_output_stream:
            return data, stream
        return data

    def get_params(self, deep=False):
        params = dict(
            num=self.N,
            seed=self.seed,
            M=self.M,
            step=self.step,
            lam=self.lam_0,
            eps=self.eps,
            nu=self.nu,
            B_thresh=self.B_thresh,
            n_thresh=self.n_thresh,
            go_fast=self.go_fast,
            copy_row_on_teleport=self.copy_row_on_teleport,
            num_grad_q=self.num_grad_q,
            sigma_orig_adjustment=self.sigma_orig_adjust,
        )

        return params | super().get_params()

    def uninitialized_copy(self):
        bw = Bubblewrap(**self.get_params())
        bw.step = self.step
        bw.log = self.log
        return bw

    def log_for_partial_fit(self, data, stream):
        if self.log_level > 0 and self.is_initialized and self.input_streams[stream] == 'X' and not numpy.isnan(data).any():
            if 'alpha' not in self.log:
                for key in ['alpha', 'entropy', 't', 'log_pred_p', 'log_pred_p_origin_t']:
                    self.log[key] = []
            if hasattr(data, 't'):
                t = data.t
            else:
                # TODO: this is not a great fix, but it works
                t = self.obs.n_obs
            self.log['alpha'].append(numpy.array(self.alpha))
            self.log['entropy'].append(self.entropy(n_steps=self.n_steps_to_predict))
            self.log['t'].append(t)

            real_time_offset = (self.dt or 1) * self.n_steps_to_predict
            self.unevaluated_predictions[t + real_time_offset] = (t, self.unevaluated_log_pred_p(self.n_steps_to_predict))
            for t_to_eval in list(self.unevaluated_predictions.keys()):
                if numpy.isclose(t, t_to_eval):
                    origin_t, f = self.unevaluated_predictions[t_to_eval]
                    self.log['log_pred_p'].append(f(data))
                    self.log['log_pred_p_origin_t'].append(origin_t)
                    del self.unevaluated_predictions[t_to_eval]

    def get_alpha_at_t(self, t, alpha=None, relative_t=False):
        alpha = alpha or self.alpha
        t = t if relative_t else t - self.last_timepoint
        n_steps = t/self.dt
        assert numpy.isclose(round(n_steps), n_steps)
        n_steps = int(n_steps)
        return numpy.array(numpy.real(alpha @ jnp.linalg.matrix_power(self.A, n_steps)))


    def show_bubbles_2d(self, ax, dim_1=0, dim_2=1, alpha_coefficient=1, n_sds=3, name_theta=45, show_names=True, n_obs_thresh=.1):
        n_obs = numpy.array(self.n_obs)

        for n in reversed(numpy.arange(self.A.shape[0])):
            color = '#ed6713'
            alpha = .4 * alpha_coefficient
            if n in self.dead_nodes:
                color = '#000000'
                alpha = 0.05 * alpha_coefficient
            self.add_2d_bubble(ax, cov=self.L[n], center=self.mu[n], n_sds=n_sds, dim_1=dim_1, dim_2=dim_2, name=n, facecolor=color, alpha=alpha, show_name=show_names, name_theta=name_theta)

        mask = numpy.ones(self.mu.shape[0], dtype=bool)
        mask[n_obs < n_obs_thresh] = False
        mask[self.dead_nodes] = False
        ax.scatter(self.mu[mask, dim_1], self.mu[mask, dim_2], c='k', zorder=10)

    def scatter_data_with_decay(self, ax, data, dim_1=0, dim_2=1, tail_length=0,):
        ax.scatter(data[:, dim_1], data[:, dim_2], s=5, color='#004cff', alpha=numpy.power(1 - self.eps, numpy.arange(data.shape[0], 0, -1)))
        if tail_length > 0:
            start = max(data.shape[0] - tail_length, 0)
            ax.plot(data[start:, 0], data[start:, 1], linewidth=3, color='#004cff', alpha=.5)

        ax.scatter(data[0, 0], data[0, 1], color="#004cff", s=10)

    def show_active_bubbles_2d(self, ax, dim_1=0, dim_2=1, name_theta=45, n_sds=3):
        to_draw = numpy.argsort(numpy.array(self.alpha))[-3:]
        opacities = numpy.array(self.alpha)[to_draw]
        opacities = opacities * .5 / opacities.max()

        for i, n in enumerate(to_draw):
            self.add_2d_bubble(ax, self.L[n], self.mu[n], n_sds=n_sds, dim_1=dim_1, dim_2=dim_2, name=n, alpha=opacities[i], name_theta=name_theta)


    def show_active_bubbles_and_connections_2d(self, ax, data, name_theta=45, n_sds=3, history_length=1):
        ax.scatter(data[:, 0], data[:, 1], s=5, color='#004cff', alpha=numpy.power(1 - self.eps, numpy.arange(data.shape[0], 0, -1)))
        # ax.scatter(data[-1, 0], data[-1, 1], s=10, color='red')

        if history_length > 1:
            start = max(data.shape[0] - history_length, 0)
            ax.plot(data[start:, 0], data[start:, 1], linewidth=3, color='#af05ed', alpha=.5)

        to_draw = numpy.argsort(numpy.array(self.alpha))[-3:]
        opacities = numpy.array(self.alpha)[to_draw]
        opacities = opacities * .5 / opacities.max()

        for i, n in enumerate(to_draw):
            self.add_2d_bubble(ax, self.L[n], self.mu[n], passed_sig=False, n_sds=n_sds, name=n, alpha=opacities[i], name_theta=name_theta)

            if i == 2:
                connections = numpy.array(self.A[n])
                self_connection = connections[n]
                other_connection = numpy.array(connections)
                other_connection[n] = 0
                c_to_draw = numpy.argsort(connections)[-3:]
                c_opacities = (other_connection / other_connection.sum())[c_to_draw]
                for j, m in enumerate(c_to_draw):
                    if n != m:
                        line = numpy.array(self.mu)[[n, m]]
                        ax.plot(line[:, 0], line[:, 1], color='k', alpha=1)

    def show_A(self, ax, show_log=False):
        A = numpy.array(self.A)
        if show_log:
            A = numpy.log(A)
        img = ax.imshow(A, aspect='equal', interpolation='nearest')
        # fig.colorbar(img)

        ax.set_title("Transition Matrix (A)")
        ax.set_xlabel("To")
        ax.set_ylabel("From")

        ax.set_xticks(numpy.arange(self.N))
        live_nodes = [x for x in numpy.arange(self.N) if x not in self.dead_nodes]
        ax.set_yticks(live_nodes)

    def show_alpha(self, ax, history_length=20, show_log=False):

        to_show = numpy.array(self.log['alpha'][-history_length:]).T

        if show_log:
            to_show = numpy.log(to_show)

        ims = ax.imshow(to_show, aspect='auto', interpolation='nearest')

        ax.set_title("State Estimate ($\\alpha$)")
        live_nodes = [x for x in numpy.arange(self.N) if x not in self.dead_nodes]
        ax.set_yticks(live_nodes)
        if len(live_nodes) > 20:
            ax.set_yticklabels([str(x) if idx % (len(live_nodes) // 20) == 0 else "" for idx, x in enumerate(live_nodes)])
        else:
            ax.set_yticklabels([str(x) for x in live_nodes])
        ax.set_ylabel("bubble")
        ax.set_xlabel("steps (ago)")


    def show_nstep_pdf(self, ax, other_axis, fig, density=50, current_location=None, offset_location=None, hmm=None, method="br", offset=1, show_colorbar=True):
        """
        the other_axis is supposed to be something showing the bubbles, so they line up
        """
        if ax.collections and show_colorbar:
            old_vmax = ax.collections[-3].colorbar.vmax
            old_vmin = ax.collections[-3].colorbar.vmin
            ax.collections[-3].colorbar.remove()

        xlim = other_axis.get_xlim()
        ylim = other_axis.get_ylim()


        x_bins = numpy.linspace(*xlim, density + 1)
        y_bins = numpy.linspace(*ylim, density + 1)
        pdf = numpy.zeros(shape=(density, density))
        for i in range(density):
            for j in range(density):
                # TODO: you could really speed this up by calculating alpha and only plotting the non-zero bubbles
                x = numpy.array([x_bins[i] + x_bins[i + 1], y_bins[j] + y_bins[j + 1]]) / 2
                b_values = self.logB_jax(x, self.mu, self.L, self.L_diag)
                pdf[i, j] = self.alpha @ numpy.linalg.matrix_power(self.A, offset) @ numpy.exp(b_values)
                # elif method == 'hmm':
                #     emission_model = hmm.emission_model
                #     node_history, _ = br.output_ds.get_history()
                #     current_node = node_history[-1]
                #     state_p_vec = numpy.zeros(emission_model.means.shape[0])
                #     state_p_vec[current_node] = 1
                #
                #     x = numpy.array([x_bins[i] + x_bins[i + 1], y_bins[j] + y_bins[j + 1]]) / 2
                #     pdf_p_vec = numpy.zeros(emission_model.means.shape[0])
                #     for k in range(pdf_p_vec.size):
                #         mu = emission_model.means[k]
                #         sigma = emission_model.covariances[k]
                #         displacement = x - mu
                #         pdf_p_vec[k] = 1 / (numpy.sqrt((2 * numpy.pi)**mu.size * numpy.linalg.det(sigma))) * numpy.exp(-1 / 2 * displacement.T @ numpy.linalg.inv(sigma) @ displacement)
                #
                #     pdf[i, j] = state_p_vec @ numpy.linalg.matrix_power(hmm.transition_matrix, offset) @ pdf_p_vec

        cmesh = ax.pcolormesh(x_bins, y_bins, pdf.T)
        if show_colorbar:
            fig.colorbar(cmesh)

        if offset_location is not None:
            ax.scatter(offset_location[0], offset_location[1], c='white')

        if current_location is not None:
            ax.scatter(current_location[0], current_location[1], c='red')

        ax.set_title(f"{offset}-step pred.")




    @classmethod
    def add_2d_bubble(cls, ax, cov, center, passed_sig=False, dim_1=0, dim_2=1, **kwargs):
        if not passed_sig:
            el = numpy.linalg.inv(cov)
            sig = el.T @ el
        else:
            sig = cov
        proj_mat = numpy.eye(sig.shape[0])[[dim_1, dim_2], :]
        sig = proj_mat @ sig @ proj_mat.T
        center = proj_mat @ center
        cls.add_2d_bubble_from_sig(ax, sig, center, **kwargs)


    @classmethod
    def add_2d_bubble_from_sig(cls, ax, sig, center, n_sds=3, facecolor='#ed6713', name=None, alpha=1., name_theta=45, show_name=True):
        assert center.size == 2
        assert sig.shape == (2,2)

        u, s, v = numpy.linalg.svd(sig)
        width, height = numpy.sqrt(s[0]) * n_sds, numpy.sqrt(s[1]) * n_sds  # note width is always bigger
        angle = atan2(v[0, 1], v[0, 0]) * 360 / (2 * numpy.pi)
        el = Ellipse((center[0], center[1]), width, height, angle=angle, zorder=8)
        el.set_alpha(alpha)
        el.set_clip_box(ax.bbox)
        el.set_facecolor(facecolor)
        ax.add_artist(el)

        if show_name:
            theta1 = name_theta - angle
            r = cls._ellipse_r(width / 2, height / 2, theta1 / 180 * numpy.pi)
            ax.text(center[0] + r * numpy.cos(name_theta / 180 * numpy.pi), center[1] + r * numpy.sin(name_theta / 180 * numpy.pi), name, clip_on=True)

    @staticmethod
    def _ellipse_r(a, b, theta):
        return a * b / numpy.sqrt((numpy.cos(theta) * b)**2 + (numpy.sin(theta) * a)**2)

    @staticmethod
    def compare_runs(bws, behavior_dict=None, t_in_samples=False):
        import matplotlib.pyplot as plt
        def _one_sided_ewma(data, com=100):
            import pandas as pd
            return pd.DataFrame(data=dict(data=data)).ewm(com).mean()["data"]

        def plot_with_trendline(ax, times, data, color, com=100):
            ax.plot(times, data, alpha=.25, color=color)
            smoothed_data = _one_sided_ewma(data, com, )
            ax.plot(times, smoothed_data, color=color)

        bws: [Bubblewrap]
        for bw in bws:
            assert bw.log_level > 0

        has_behavior = behavior_dict is not None


        fig, axs = plt.subplots(figsize=(14, 5), nrows=2 + has_behavior, ncols=2, sharex='col', layout='tight',
                                gridspec_kw={'width_ratios': [7, 1]})

        common_time_start = max([min(bw.log['t']) for bw in bws])
        common_time_end = min([max(bw.log['t']) for bw in bws])
        halfway_time = (common_time_start + common_time_end) / 2

        to_write = [[] for _ in range(axs.shape[0])]
        colors = ['C0'] + ['k'] * (len(bws) - 1)
        for idx, bw in enumerate(bws):
            color = colors[idx]

            # plot prediction
            t = numpy.array(bw.log['log_pred_p_origin_t'])
            t_to_plot = t
            if t_in_samples:
                t_to_plot = t / bw.dt
            to_plot = numpy.array(bw.log['log_pred_p'])
            plot_with_trendline(axs[0, 0], t_to_plot, to_plot, color)
            last_half_mean = to_plot[(halfway_time < t) & (t < common_time_end)].mean()
            to_write[0].append((idx, f'{last_half_mean:.2f}', {'color': color}))
            axs[0, 0].set_ylabel('log pred. p')

            # plot entropy
            t = numpy.array(bw.log['t'])
            t_to_plot = t
            if t_in_samples:
                t_to_plot = t / bw.dt
            to_plot = numpy.array(bw.log['entropy'])
            plot_with_trendline(axs[1, 0], t_to_plot, to_plot, color)
            last_half_mean = to_plot[(halfway_time < t) & (t < common_time_end)].mean()
            to_write[1].append((idx, f'{last_half_mean:.2f}', {'color': color}))
            axs[1, 0].set_ylabel('entropy')

            max_entropy = numpy.log2(bw.N)
            axs[1, 0].axhline(max_entropy, color='k', linestyle='--')

            # plot behavior
            if has_behavior:
                from adaptive_latents.utils import resample_matched_timeseries

                t = behavior_dict[idx]['predicted_behavior_t']
                targets = resample_matched_timeseries(
                    behavior_dict[idx]['true_behavior'],
                    behavior_dict[idx]['true_behavior_t'],
                    t
                )
                estimates = behavior_dict[idx]['predicted_behavior']

                test_s = t > (t[0] + t[-1]) / 2

                correlations = [numpy.corrcoef(estimates[test_s, i], targets[test_s, i])[0, 1] for i in range(estimates.shape[1])]
                corr_str = '\n'.join([f'{r:.2f}' for r in correlations] )
                to_write[2].append((idx, corr_str, {'fontsize': 'x-small'}))

                t_to_plot = t
                if t_in_samples:
                    t_to_plot = t / bw.dt
                for i in range(estimates.shape[1]):
                    axs[2,0].plot(t_to_plot, targets[:, i], color=f'C{i}')
                    axs[2,0].plot(t_to_plot, estimates[:, i], color=f'C{i}', alpha=.5)
                # axs[2,0].axvline(t[test_s].min(), color='k')
                axs[2,0].set_xlabel("time")
                axs[2,0].set_ylabel("behavior")


        # this sets the axis bounds for the text
        for axis in axs[:, 0]:
            data_lim = numpy.array(axis.dataLim).T.flatten()
            bounds = data_lim
            bounds[:2] = (bounds[:2] - bounds[:2].mean()) * numpy.array([1.02, 1.2]) + bounds[:2].mean()
            bounds[2:] = (bounds[2:] - bounds[2:].mean()) * numpy.array([1.05, 1.05]) + bounds[2:].mean()
            axis.axis(bounds)
            axis.format_coord = lambda x, y: 'x={:g}, y={:g}'.format(x, y)

        # this prints the last-half means
        for i, l in enumerate(to_write):
            for idx, text, kw in l:
                x, y = .92, .93 - .1 * idx
                x, y = axs[i, 0].transLimits.inverted().transform([x, y])
                axs[i, 0].text(x, y, text, clip_on=True, verticalalignment='top', **kw)

        # this creates the axis for the parameters
        gs = axs[0, 1].get_gridspec()
        for a in axs[:, 1]:
            a.remove()
        axbig = fig.add_subplot(gs[:, 1])
        axbig.axis("off")

        # this generates and prints the parameters
        params_per_bw_list = [bw.get_params() for bw in bws]
        super_param_dict = {}
        for key in params_per_bw_list[0].keys():
            values = [p[key] for p in params_per_bw_list]
            if len(set(values)) == 1:
                values = values[0]
                if key in {'input_streams', 'output_streams', 'log_level'}:
                    continue
            super_param_dict[key] = values
        to_write = "\n".join(f"{k}: {v}" for k, v in super_param_dict.items())
        axbig.text(0, 1, to_write, transform=axbig.transAxes, verticalalignment="top")

