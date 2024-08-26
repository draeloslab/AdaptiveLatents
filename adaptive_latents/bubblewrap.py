import numpy
import jax.numpy as jnp
from collections import deque
from jax import jit, vmap
from jax import nn, random
import jax
import warnings
from adaptive_latents.config import use_config_defaults
from adaptive_latents.transformer import TransformerMixin
from adaptive_latents.timed_data_source import ArrayWithTime

# todo: make this a parameter?
epsilon = 1e-10


class BaseBubblewrap:
    @use_config_defaults
    # note the defaults in this signature are overridden by the defaults in adaptive_latents_config
    def __init__(self, dim, num=1000, seed=42, M=30, lam=1, nu=1e-2, eps=3e-2, B_thresh=1e-4, step=1e-6, n_thresh=5e-4, batch=False, batch_size=1, go_fast=False, copy_row_on_teleport=True, num_grad_q=1, sigma_orig_adjustment=0):

        self.N = num  # Number of nodes
        self.d = dim  # dimension of the space
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

        self.batch = batch
        self.batch_size = batch_size
        if not self.batch:
            self.batch_size = 1

        self.go_fast = go_fast

        self.key = random.PRNGKey(self.seed)
        numpy.random.seed(self.seed)
        # TODO: change this to use the `rng` system

        # observations of the data; M is how many to keep in history
        if self.batch:
            M = self.batch_size
        self.obs = Observations(self.d, M=M, go_fast=go_fast)
        self._add_jited_functions()
        self.mu_orig = None

        self.is_initialized = False
        self.frozen = False

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
        if self.batch and not self.go_fast:
            var = self.obs.cov
        else:
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
        self.pred_ahead = jit(pred_ahead, static_argnames=['steps_ahead'])
        self.sum_me = jit(sum_me)
        self.compute_L = jit(vmap(get_L, (0, 0)))
        self.get_amax = jit(amax)
        self.get_entropy = jit(entropy, static_argnames=['steps_ahead'])

    def observe(self, x):
        # Get new data point and update observation history

        ## Do all observations, and then update mu0, sigma0
        if self.batch:
            for i in range(len(x)):  # x array of observations
                self.obs.new_obs(x[i])
        else:
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
        if self.batch:
            for o in self.obs.saved_obs:
                self.single_e_step(o)
        else:
            self.single_e_step(self.obs.curr)

    def single_e_step(self, x):
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

    def freeze(self):
        self.frozen = True
        self.obs.freeze()

    def __getstate__(self):
        return _unjax_state(self)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not state["frozen"]:
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
        batch=False,
        batch_size=1,
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
    def __init__(self, dim, M=5, go_fast=True):
        self.M = M  # how many observed points to hold in memory
        self.d = dim  # dimension of coordinate system
        self.go_fast = go_fast

        self.curr = None
        self.saved_obs = deque(maxlen=self.M)

        self.mean = None
        self.last_mean = None

        self.cov = None

        self.n_obs = 0

        self.frozen = False

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

    def freeze(self):
        self.frozen = True

    def __getstate__(self):
        return _unjax_state(self)


def _unjax_state(self):
    to_save = {}
    _pickle_changes = []
    for key, value in self.__dict__.items():
        if callable(value) and "jit" in str(value):
            _pickle_changes.append((key, "callable"))
            continue

        elif self.frozen and "jax" in str(type(value)) and "Array" in str(type(value)):
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


class Bubblewrap(TransformerMixin, BaseBubblewrap):
    def partial_fit_transform(self, data, stream=0, return_output_stream=False):
        # partial fit
        if self.input_streams[stream] == 'X' and not numpy.isnan(data).any():
            output = []
            for i in range(len(data)):
                self.observe(data[i])

                if self.is_initialized:
                    self.e_step()
                    self.grad_Q()
                elif self.obs.n_obs > self.M:
                    self.init_nodes()

                # transform
                if self.is_initialized:
                    o = numpy.array(self.alpha)
                else:
                    o = numpy.zeros(self.N) * numpy.nan
                output.append(o)

            if isinstance(data, ArrayWithTime):
                data = ArrayWithTime(output, data.t)

        stream = self.output_streams[stream]
        if return_output_stream:
            return data, stream
        return data


    def partial_fit(self, data, stream=0):
        raise NotImplementedError()

    def transform(self, data, stream=0, return_output_stream=False):
        raise NotImplementedError()

    def freeze(self, b=True):
        raise NotImplementedError()