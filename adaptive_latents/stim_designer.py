import time
import functools

from adaptive_latents.input_sources.autoregressor import AdamOptimizer
import jax
import numpy
import jax.numpy as jnp
from jax.nn import relu
import warnings
from collections import deque

from jax import grad, jit, value_and_grad
from jaxopt import ScipyBoundedMinimize

class StimDesigner:
    def __init__(
            self,
            max_l0_norm=30,
            rng_seed=0,
            should_log=False,
            lam_1=0.001,
    ):
        self.rng_seed = rng_seed
        self.rng = numpy.random.default_rng(rng_seed)
        self.max_l0_norm = max_l0_norm
        self.should_log = should_log
        self.lam_1 = lam_1
        self.log = []


    def design_stim(self, v, u_dimension, u_to_s_function=None):
        start_time = time.time()
        assert len(v.shape) == 2
        assert self.max_l0_norm > 0


        u = self.rng.uniform(size=(u_dimension,)) * .1

        lb = jnp.zeros_like(u)
        ub = jnp.ones_like(u)
        bounds = (lb, ub)

        def objective(u):
            s = u_to_s_function(u)
            s_norm = jnp.linalg.norm(s)
            loss = self.lam_1 * (self.max_l0_norm - jnp.sum(jnp.abs(u)))
            loss += jnp.dot(s, v) / (s_norm + 1e-10)
            return -loss.reshape()

        runner = ScipyBoundedMinimize(fun=objective, method='l-bfgs-b')
        result = runner.run(u, bounds=bounds)
        u = numpy.array(result.params)

        if u.max() > 0:
            u = numpy.array(u / u.max())

        unthresholded_u = numpy.array(u)
        unthresholded_s = u_to_s_function(unthresholded_u)
        idx = numpy.argsort(u)
        u[idx[:-self.max_l0_norm]] = 0

        if self.should_log:
            self.log.append({
                'time': time.time() - start_time,
                'v':v,
                'u':u,
                's':u_to_s_function(u),
                'unthresholded_u': unthresholded_u,
                'unthresholded_s': unthresholded_s,
                # 'u_to_s_function_is_none':u_to_s_function is None,
                # 'loss_history':loss_history,
                # 'lam_1_history':lam_1_history,
                # 'l0_history':l0_history,
                # 's_history':s_history,
            })
        return u, u_to_s_function(u)