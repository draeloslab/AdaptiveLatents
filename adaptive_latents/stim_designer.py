import time
import numpy
import jax.numpy as jnp
from jaxopt import ScipyBoundedMinimize
import itertools

class StimDesigner:
    def __init__(
            self,
            max_l0_norm=30,
            rng_seed=0,  # TODO: make this an rng
            should_log=False,
            lam_1=0.001,
            inter_stim_interval_generator=None,
            optimization_method='jaxopt',
            stim_timing_method='regular',
            initial_nostim_period=1,
            u_to_s_model_type='identity',
            n_identity_initialization=1,
    ):
        self.rng_seed = rng_seed
        self.rng = numpy.random.default_rng(rng_seed)
        assert max_l0_norm > 0
        self.max_l0_norm = max_l0_norm
        self.should_log = should_log
        self.lam_1 = lam_1

        self.optimization_method = optimization_method
        self.u_to_s_model_type = u_to_s_model_type
        self.n_identity_initialization = n_identity_initialization
        self.stim_timing_method = stim_timing_method
        self.initial_nostim_period = initial_nostim_period

        if inter_stim_interval_generator is None:
            inter_stim_interval_generator = itertools.repeat(1)
        self.inter_stim_interval_generator = inter_stim_interval_generator
        self.last_stim_time = None
        self.current_isi = None

        self.log = []

        self.objective_history = []

    def stim_when_extreme(self, current_t, objective_value):
        self.objective_history.append(objective_value)
        return current_t > 50 and objective_value == numpy.nanmin(self.objective_history)

    def decide_whether_to_stim(self, current_t, **kwargs):
        if current_t < self.initial_nostim_period:
            return False

        if self.stim_timing_method == 'isi':  # or 'regular'
            if self.last_stim_time is None:
                self.last_stim_time = self.initial_nostim_period if self.initial_nostim_period is not None else 0
                self.current_isi = next(self.inter_stim_interval_generator)
            if current_t > self.last_stim_time + self.current_isi:
                self.last_stim_time = current_t
                self.current_isi = next(self.inter_stim_interval_generator)
                return True
            return False
        elif self.stim_timing_method == 'extreme':
            return self.stim_when_extreme(current_t, **kwargs)
        elif self.stim_timing_method == 'random':
            return kwargs['stim_time_rng'].random() < 1/next(self.inter_stim_interval_generator) * kwargs['input_array_dt']
        else:
            raise ValueError()


    @staticmethod
    def desired_stim_direction(equivalent_projection_matrix, stim_direction_type, rng):  # TODO: use built-in rng
        if stim_direction_type == 'first':
            desired_stim = numpy.zeros((equivalent_projection_matrix.shape[1], 1))
            desired_stim[0] = 1
        elif stim_direction_type == 'first2':
            desired_stim = numpy.zeros((equivalent_projection_matrix.shape[1], 2))
            desired_stim[0] = 1
            desired_stim[1] = 1
        elif stim_direction_type == 'col':
            desired_stim = numpy.zeros((equivalent_projection_matrix.shape[1], 1))
            desired_stim[rng.choice(equivalent_projection_matrix.shape[1]), 0] = 1
        elif stim_direction_type == 'random':
            desired_stim = rng.normal(size=(equivalent_projection_matrix.shape[1], 1))
            desired_stim = desired_stim / numpy.linalg.norm(desired_stim)
        else:
            raise ValueError()
        return desired_stim

    def register_stim(self):
        pass

    def design_stim_jaxopt(self, v, u_dimension, u_to_s_function=None):
        if u_to_s_function is None:
            u_to_s_function = lambda x: x

        u = self.rng.uniform(size=(u_dimension,)) * .1

        lb = jnp.zeros_like(u)
        ub = jnp.ones_like(u)
        bounds = (lb, ub)

        def objective(u):
            s = u_to_s_function(u)
            s_norm = jnp.linalg.norm(s)
            loss = self.lam_1 * (self.max_l0_norm - jnp.sum(jnp.abs(u)))
            loss += jnp.dot(s, v) / (s_norm + 1e-10)
            # new: loss += jnp.linalg.norm(jnp.dot(s, v))**2 / (s_norm + 1e-10)
            return -loss.reshape()

        runner = ScipyBoundedMinimize(fun=objective, method='l-bfgs-b')
        result = runner.run(u, bounds=bounds)
        u = numpy.array(result.params)

        if u.max() > 0:
            u = numpy.array(u / u.max())


        idx = numpy.argsort(u)
        u[idx[:-self.max_l0_norm]] = 0

        return u, {'s': u_to_s_function(u)}


    @staticmethod
    def design_stim_cheat(v, equivalent_projection_matrix):
        return (equivalent_projection_matrix @ v).flatten(), {'s': numpy.nan * v}
        # TODO: delete s here, it's mostly for compatibility with an old version of sim_stim


    def design_stim(self, v, **kwargs):
        start_time = time.time()
        assert len(v.shape) == 2

        match self.optimization_method:
            case 'jaxopt':
                u, l = self.design_stim_jaxopt(v, kwargs['u_dimension'], kwargs['u_to_s_function'])
            case 'cheat':
                u, l = self.design_stim_cheat(v, kwargs['equivalent_projection_matrix'])
            case _:
                raise ValueError()


        if self.should_log:
            self.log.append({
                'optimization_time': time.time() - start_time,
                'v':v,
                'u':u,
            } | l)

        return u
