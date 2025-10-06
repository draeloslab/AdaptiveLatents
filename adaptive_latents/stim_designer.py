import time
import numpy
import jax.numpy as jnp
from jaxopt import ScipyBoundedMinimize
import itertools

import jax
import os, json, time


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
            u_to_s_model_type='identity', # TODO: remove
            n_identity_initialization=1,
    ):
        self.rng_seed = rng_seed
        self.rng = numpy.random.default_rng(rng_seed)
        assert max_l0_norm > 0
        self.max_l0_norm = max_l0_norm
        self.should_log = should_log
        self.lam_1 = lam_1

        self.optimization_method = optimization_method
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

        # changed the bounds
        u = self.rng.uniform(size=(u_dimension,)) * 0.2 - 0.1
        lb = -jnp.zeros_like(u)
        ub = jnp.ones_like(u)
        bounds = (lb, ub)

        log_path = os.path.expanduser(r"C:\Users\secom\OneDrive\Documents\DraelosLab\AdaptiveLatents\AdaptiveLatents\workspace\Alexworkspace\metrics_data.jsonl")  # or "~/stim_logs/stim_history.jsonl"
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
       
        ep = 1e-10

        v_flat = jnp.ravel(jnp.asarray(v))   # flatten for now 
        v_norm = jnp.linalg.norm(v_flat) + ep
       
        eval_counter = {"k": 0} # eval steps per obj call

        def _collector(obj, cos_sim, angle_deg, align_proj):
            k = eval_counter["k"]
            rec = {
                "eval": int(k),
                "obj": float(obj),
                "align_cos": float(cos_sim),
                "angle_deg": float(angle_deg),
                "align_proj": float(align_proj),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            eval_counter["k"] = k + 1


        def objective(u):

            s = u_to_s_function(u)
            s_flat = jnp.ravel(s)
            s_norm = jnp.linalg.norm(s_flat) + ep

            dot_sv = jnp.dot(s_flat, v_flat) 
            cos_sim = dot_sv / (s_norm * v_norm)
            # should i get rid of v part for cosine sim as well?
            # getting rid of v adds the potential for v to be a matrix
            # since you cannot take regular norm from matrix
            # frobenius norm?
            cos_sim = jnp.clip(cos_sim, -1.0, 1.0)
            angle_deg = jnp.degrees(jnp.arccos(cos_sim))

            # loss = -cos_sim + self.lam_1 * (jnp.sum(jnp.abs(u))) # old

            loss = -cos_sim + self.lam_1 * (self.max_l0_norm - jnp.sum(jnp.abs(u))) # old

            # loss += jnp.linalg.norm(jnp.dot(s, v))**2 / (s_norm + 1e-10)
            align_proj = dot_sv**2 / (s_norm)
            obj = loss
            # why not loss += jnp.dot(s, v) / ((jnp.linalg.norm(s) + 1e-10) * (jnp.linalg.norm(v) + 1e-10))

            jax.debug.callback(
            _collector,
            jnp.squeeze(obj),
            jnp.squeeze(cos_sim),
            jnp.squeeze(angle_deg),
            jnp.squeeze(align_proj),
            )

            return jnp.squeeze(loss)

        runner = ScipyBoundedMinimize(fun=objective, method='l-bfgs-b', 
                                    #   maxiter=2000, options={'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
                                      ) 
        result = runner.run(u, bounds=bounds)
        u = numpy.array(result.params)

        if u.max() > 0:
            u = numpy.array(u / u.max())


        idx = numpy.argsort(u)
        u[idx[:-self.max_l0_norm]] = 0


        return u, {'s': u_to_s_function(u)}



    def design_stim(self, v, **kwargs):
        start_time = time.time()
        assert len(v.shape) == 2

        l = {}
        match self.optimization_method:
            case 'jaxopt':
                u, l = self.design_stim_jaxopt(v, kwargs['u_dimension'], kwargs['u_to_s_function'])
            case 'cheat_lowd_vec':
                u = (kwargs['equivalent_projection_matrix'] @ v).flatten(),
            case 'cheat_highd_vec_single_neurons':
                u = numpy.zeros(kwargs['equivalent_projection_matrix'].shape[0])
                u[self.rng.choice(kwargs['equivalent_projection_matrix'].shape[0])] = 1
            case 'cheat_highd_vec_many_neurons':
                u = numpy.zeros(kwargs['equivalent_projection_matrix'].shape[0])
                u[self.rng.choice(kwargs['equivalent_projection_matrix'].shape[0], size=self.max_l0_norm, replace=False)] = 1
            case _:
                raise ValueError()


        if self.should_log:
            self.log.append({
                'optimization_time': time.time() - start_time,
                'v':v,
                'u':u,
                's': numpy.nan * v
            } | l)

        return u
