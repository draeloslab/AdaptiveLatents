import time
import numpy
import jax.numpy as jnp
from jaxopt import ScipyBoundedMinimize
import itertools

import jax
import os, json, time

from functools import partial
from jaxopt import LBFGS

import time

import uuid
from pathlib import Path


class StimDesigner:
    def __init__(
            self,
            max_l0_norm=30,
            rng_seed=None,  # TODO: make this an rng
            should_log=False,
            lam_1=0.001, # 0.001 before
            #0.000001 best
            inter_stim_interval_generator=None,
            optimization_method='jaxopt',
            stim_timing_method='regular',
            initial_nostim_period=1,
            u_to_s_model_type='identity', # TODO: remove
            n_identity_initialization=1,
            session_id=None,      
            script_run_id=None,       
            log_dir=None,
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

        self.session_id = session_id or "session_default"
        self.script_run_id = script_run_id or uuid.uuid4().hex

        base_dir = Path(log_dir) if log_dir is not None else Path(".")
        base_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = base_dir / f"metrics_{self.session_id}.jsonl"

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

    def design_stim_jaxopt(self, v, u_dimension, u_to_s_function=None, init_u0=None):
       
        run_id = uuid.uuid4().hex
       
        if u_to_s_function is None:
            u_to_s_function = lambda x: x


        # stage_maxiter = 5     # ~ how many steps before it plateaus 
        # eval_cap      = 5
        # ls_max        = 5
        stage_maxiter = eval_cap = ls_max = 10
        perturb_scale = 0.8 # tiny nudge after each stage

        # changed the bounds
        #u = self.rng.uniform(size=(u_dimension,)) * 0.2
        u = jnp.asarray(init_u0) if init_u0 is not None else self.rng.uniform(size=(u_dimension,)) * 0.1

        lb = jnp.zeros_like(u)
        ub = jnp.ones_like(u)
        bounds = (lb, ub)

        #log_path = os.path.expanduser(r"C:\Users\secom\OneDrive\Documents\DraelosLab\AdaptiveLatents\AdaptiveLatents\workspace\Alexworkspace\metrics_data.jsonl")  
        #os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
       
        log_buffer = []

        eval_counter = {"k": 0} # eval steps per obj call

        def _collector(obj, cos_sim, angle_deg, align_proj, regtrack):
            k = eval_counter["k"]
            angle_history.append(float(angle_deg))
            rec = {
                "session_id" : self.session_id,
                "script_run_id" : self.script_run_id,
                "run_id": run_id,
                "optimizer" : "jaxopt",
                "eval": int(k),
                "obj": float(obj),
                "align_cos": float(cos_sim),
                "angle_deg": float(angle_deg),
                "align_proj": float(align_proj),
                "reg term": float(regtrack),
            }
            log_buffer.append(rec)
            eval_counter["k"] = k + 1
        angle_history = []
        #v = jnp.maximum(jnp.ravel(v), 0.0)
        v = jnp.ravel(jnp.asarray(v))
        
        def objective(u):

            s = u_to_s_function(u)
            s = jnp.ravel(s)
            s_norm = jnp.linalg.norm(s) + 1e-10


            dot_sv = jnp.dot(s, v) 
            cos_sim = dot_sv / (s_norm)

            # if v is matrix, how would u compare s(u) and v in the first place?

            cos_sim_clipped = jnp.clip(cos_sim, -1.0, 1.0)
            angle_deg = jnp.degrees(jnp.arccos(cos_sim_clipped))

            # loss = -cos_sim + self.lam_1 * (jnp.sum(jnp.abs(u))) # old
            regterm = self.lam_1 * (self.max_l0_norm - jnp.sum(jnp.abs(u)))
                
            loss = -(cos_sim) + regterm 
            regtrack = regterm
            
            align_proj = dot_sv**2 / (s_norm)
            # why not loss += jnp.dot(s, v) / ((jnp.linalg.norm(s) + 1e-10) * (jnp.linalg.norm(v) + 1e-10))

            jax.debug.callback(
            _collector,
            jnp.squeeze(loss),
            jnp.squeeze(cos_sim_clipped), # cos term
            jnp.squeeze(angle_deg),
            jnp.squeeze(align_proj),
            jnp.squeeze(regtrack) #reg term
            )

            return jnp.squeeze(loss)


        runner = ScipyBoundedMinimize(fun=objective, method='l-bfgs-b', 
                                    #maxiter=stage_maxiter, 
                                    maxiter= 500,
                                    options={
                                    # "maxfun": eval_cap, "maxls": ls_max
                                    "maxfun": 20000,
                                    "gtol": 1e-12,   # gradient tolerance
                                    "ftol": 1e-12,   # function change tolerance
                                    "maxls": 50, 
                                    }
                                    ) 
        start_opt = time.perf_counter()
        result = runner.run(u, bounds=bounds)
        end_opt = time.perf_counter()
        timeforopt = end_opt - start_opt

        u = numpy.array(result.params)

        # u1 = numpy.clip(u + self.rng.normal(size=u_dimension) * perturb_scale, 0.0, 0.2)

        # runner2 = ScipyBoundedMinimize(
        # fun=objective,
        # method="l-bfgs-b",
        # maxiter=stage_maxiter,
        # options={"maxfun": eval_cap, "maxls": ls_max},
        # )
        # res2 = runner2.run(u1, bounds=bounds)
        # u = numpy.array(res2.params)

        # look into it
        if u.max() > 0:
            u = numpy.array(u / u.max())
        # 30 neuron
        idx = numpy.argsort(u)
        u[idx[:-self.max_l0_norm]] = 0

        final_angle = float(angle_history[-1]) if len(angle_history) else None
        #iterations = len(angle_history)
        nnz = int((u > 1e-6).sum())

        summary_record = {
            "session_id" : self.session_id,
            "script_run_id" : self.script_run_id,
            "run_id": run_id,
            "optimizer": "lbfgs",
            #"scenario": scenario,
            #"seed": int(self.rng_seed if self.rng_seed is not None else -1),
            "total_time": float(timeforopt),
            #"iterations": iterations,
            "final_angle_deg": final_angle,
            "nnz": nnz,
            "success_lt_90": final_angle < 90 if final_angle is not None else None,
            "success_lt_45": final_angle < 45 if final_angle is not None else None,
        }
        log_buffer.append(summary_record)

        with open(self.log_path, "a", encoding="utf-8") as f:
            for rec in log_buffer:
                f.write(json.dumps(rec) + "\n")

        return u, {'s': u_to_s_function(u)}

    def design_stim_cem(self, v, u_dimension, u_to_s_function=None,
                    iters=20, pop=256, elite_frac=0.1, init_std=1.0, init_u0=None):
        if u_to_s_function is None:
            u_to_s_function = lambda x: x

        # ---- shared pieces (unit v, collector, objective) ----
        ep    = 1e-10
        # v_vec = jnp.ravel(jnp.asarray(v))
        # v_unit = v_vec / (jnp.linalg.norm(v_vec) + ep)

        log_path = os.path.expanduser(
            r"C:\Users\secom\OneDrive\Documents\DraelosLab\AdaptiveLatents\AdaptiveLatents\workspace\Alexworkspace\metrics_data.jsonl"
        )
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        eval_counter = {"k": 0}

        def _collector(obj, cos_sim, angle_deg, align_proj, regtrack):
            k = eval_counter["k"]
            rec = {
                "eval": int(k),
                "obj": float(obj),
                "align_cos": float(cos_sim),
                "angle_deg": float(angle_deg),
                "align_proj": float(align_proj),
                "reg term": float(regtrack),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            eval_counter["k"] = k + 1

        def objective(u):
            s = jnp.ravel(u_to_s_function(u))
            s_norm = jnp.linalg.norm(s) + ep
            cos_sim = jnp.clip(jnp.dot(s, v) / s_norm, -1.0, 1.0)

            # Smooth, sign-invariant alignment + simple L1 (or hinge if you prefer)
            align_term = -(cos_sim * cos_sim)                 # ∈ [-1, 0]
            regterm = self.lam_1 * (self.max_l0_norm - jnp.sum(jnp.abs(u))) # old
            loss       = align_term + regterm

            angle_deg  = jnp.degrees(jnp.arccos(jnp.abs(cos_sim)))
            align_proj = (jnp.dot(s, v)**2) / (s_norm + ep)

            jax.debug.callback(_collector,
                            jnp.squeeze(loss),
                            jnp.squeeze(cos_sim),
                            jnp.squeeze(angle_deg),
                            jnp.squeeze(align_proj),
                            jnp.squeeze(regterm))
            return jnp.squeeze(loss)

        # ---- CEM over w; u = sigmoid(w) ∈ (0,1) ----
        rng = self.rng
        elite = max(1, int(pop * elite_frac))
        mean  = numpy.zeros(u_dimension, dtype=float)
        std   = numpy.ones(u_dimension, dtype=float) * init_std

        best_val = None
        best_u   = None

        for _ in range(iters):
            W = rng.normal(loc=mean, scale=std, size=(pop, u_dimension))
            U = 1.0 / (1.0 + numpy.exp(-W))  # sigmoid → (0,1)

            vals = []
            for i in range(pop):
                u = jnp.asarray(U[i])
                vals.append(float(objective(u)))
            vals = numpy.asarray(vals)

            idx = numpy.argsort(vals)[:elite]
            elites = W[idx]

            mean = elites.mean(axis=0)
            std  = elites.std(axis=0) + 1e-6   # keep exploration alive

            if best_val is None or vals[idx[0]] < best_val:
                best_val = vals[idx[0]]
                best_u   = 1.0 / (1.0 + numpy.exp(-mean))  # mean in w-space → u

        # Optional: polish with your existing LBFGS-B starting at best_u
        # (keeps collector through `objective`)
        # from jaxopt import ScipyBoundedMinimize
        # lb = jnp.zeros_like(best_u); ub = jnp.ones_like(best_u)
        # runner = ScipyBoundedMinimize(fun=objective, method='l-bfgs-b')
        # res = runner.run(jnp.asarray(best_u), bounds=(lb, ub))
        # u_opt = numpy.array(res.params)
        u_opt = best_u.astype(float)

        # Post-process: top-K hard prune then renormalize (same as your code)
        if u_opt.max() > 0:
            u_opt = u_opt / u_opt.max()
        idx = numpy.argsort(u_opt)
        u_opt[idx[:-self.max_l0_norm]] = 0

        return u_opt, {'s': u_to_s_function(jnp.asarray(u_opt))}

    def design_stim_admm(
        self,
        v,
        u_dimension,
        u_to_s_function=None,
        init_u0=None,
        rho=1.0,  # sweep values
        lam=None,
        max_admm_iters=30, # migth reduce
        u_inner_steps=10, # might reduce
        lr=None,
        #use_lbfgs=False,
    ):
    

        run_id = uuid.uuid4().hex

        if u_to_s_function is None:
            u_to_s_function = lambda x: x
        if lam is None:
            lam = float(self.lam_1)

        eps = 1e-10
        v_vec = jnp.ravel(jnp.asarray(v)) 

        #log_path = os.path.expanduser(
        #    r"C:\Users\secom\OneDrive\Documents\DraelosLab\AdaptiveLatents\AdaptiveLatents\workspace\Alexworkspace\metrics_data.jsonl"
        #)
        #os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log_buffer = []

        def log_row(obj, cos_sim, angle_deg, align_proj, reg_term, k):
            rec = {
                "session_id": self.session_id,
                "script_run_id": self.script_run_id,
                "run_id": run_id,
                "optimizer" : "admm",
                "eval": int(k),
                "obj": float(obj),
                "align_cos": float(cos_sim),
                "angle_deg": float(angle_deg),
                "align_proj": float(align_proj),
                "reg term": float(reg_term),
                "ts": time.time(),
            }
            log_buffer.append(rec)

        # objective portion
        def s_from(u):
            return jnp.ravel(u_to_s_function(u))

        def cos_term(u):
            s = s_from(u)
            s_norm = jnp.linalg.norm(s) + eps
            return jnp.dot(v_vec, s) / s_norm

        def f(u):
            return -cos_term(u)

        # jitted for faster computational times
        @jax.jit       
        def z_prox(w):
            return jnp.clip(w + lam / rho, 0.0, 1.0)

        if init_u0 is None:
            u0 = self.rng.uniform(size=(u_dimension,)) * 0.2
        else:
            u0 = numpy.asarray(init_u0, dtype=float)

        u = jnp.asarray(u0)
        z = jnp.clip(u, 0.0, 1.0)
        y = jnp.zeros_like(u)

        # u-subproblem: phi(u; v_k) = f(u) + (rho/2)||u - v_k||^2
        def phi(u, vk):
            return f(u) + 0.5 * rho * jnp.sum((u - vk) ** 2)

        # if use_lbfgs:
        #     def u_step(vk, u_init):
        #         solver = LBFGS(fun=partial(phi, vk=vk),
        #                         maxiter=u_inner_steps,
        #                         tol=1e-8)
        #         res = solver.run(u_init)
        #         return res.params
        # else:
        #@jax.jit
        def u_step(u_init, vk):
            u_curr = u_init
            step = lr if lr is not None else 1.0 / (rho + 1.0)
            for _ in range(u_inner_steps):
                g = jax.grad(phi)(u_curr, vk)
                u_curr = u_curr - step * g
            return u_curr
        # this jax jit version gives worse solutions, adjust the u steps and lr params for better solutions
        # its very fast tho!
        # def u_step(u_init, vk, u_steps=5, lr=0.1):
        #     def body_fun(i, u_curr):
        #         g = jax.grad(phi)(u_curr, vk)
        #         return u_curr - lr * g

        #     return jax.lax.fori_loop(0, u_steps, body_fun, u_init)

        s_init = s_from(z)
        s_norm_init = float(jnp.linalg.norm(s_init) + eps)
        dot_v_s_init = float(jnp.dot(v_vec, s_init))
        cos_init = dot_v_s_init / s_norm_init
        cos_init_clipped = float(jnp.clip(cos_init, -1.0, 1.0))
        angle_init_deg = float(numpy.degrees(numpy.arccos(cos_init_clipped)))
        align_init = float((dot_v_s_init ** 2) / s_norm_init)
        reg_init = float(lam * (self.max_l0_norm - float(jnp.sum(z))))
        obj_init = float(-cos_init + reg_init)
        log_row(obj_init, cos_init_clipped, angle_init_deg, align_init, reg_init, k=0)

        # actual admm loop
        z_prev = z
        start_opt = time.perf_counter()

        last_angle = None  # for summary later

        for k in range(1, max_admm_iters+1):
            # u-update
            vk = z - y
            u = u_step(u, vk)

            # z-update (prox of g)
            w = u + y
            z = z_prox(w)

            # dual-update
            y = y + (u - z)

            # logging from feasible z
            s_z = s_from(z)
            s_norm_val = float(jnp.linalg.norm(s_z) + eps)
            dot_v_s = float(jnp.dot(v_vec, s_z))
            cos_z = dot_v_s / s_norm_val
            cos_z_clipped = float(jnp.clip(cos_z, -1.0, 1.0))
            angle_deg = float(numpy.degrees(numpy.arccos(cos_z_clipped)))
            align_proj = float((dot_v_s ** 2) / s_norm_val)
            reg_val = float(lam * (self.max_l0_norm - float(jnp.sum(z))))
            obj_val = float(-cos_z + reg_val)

            last_angle = angle_deg
            log_row(obj_val, cos_z_clipped, angle_deg, align_proj, reg_val, k)

            # residuals
            r_norm = float(jnp.linalg.norm(u - z))              # primal
            s_norm = float(rho * jnp.linalg.norm(z - z_prev))   # dual
            z_prev = z
            if r_norm < 1e-6 and s_norm < 1e-6:
                break

        end_opt = time.perf_counter()
        timeforopt = end_opt - start_opt

        u_final = numpy.array(z)
        if u_final.max() > 0:
            u_final = u_final / u_final.max()
        idx = numpy.argsort(u_final)
        u_final[idx[:-self.max_l0_norm]] = 0.0

        nnz = int((u_final > 1e-6).sum())

        # recompute final angle from u_final 
        s_final = numpy.ravel(numpy.array(u_to_s_function(jnp.asarray(u_final))))
        s_norm_final = numpy.linalg.norm(s_final) + eps
        dot_v_s_final = float(numpy.dot(numpy.array(v_vec), s_final))
        cos_final = dot_v_s_final / s_norm_final
        cos_final_clipped = float(jnp.clip(cos_final, -1.0, 1.0))
        final_angle_deg = float(numpy.degrees(numpy.arccos(cos_final_clipped)))

        summary_record = {
            "session_id": self.session_id,
            "script_run_id": self.script_run_id,
            "run_id": run_id,
            "optimizer": "admm",
            "total_time": float(timeforopt),
            "final_angle_deg": final_angle_deg,
            "nnz": nnz,
            "success_lt_90": final_angle_deg < 90.0,
            "success_lt_45": final_angle_deg < 45.0,
        }
        log_buffer.append(summary_record)
        with open(self.log_path, "a", encoding="utf-8") as f:
            for rec in log_buffer:
                f.write(json.dumps(rec) + "\n")

        return u_final, {'s': u_to_s_function(jnp.asarray(u_final))}



    def design_stim(self, v, **kwargs):
        start_time = time.time()
        assert len(v.shape) == 2
        init_u = kwargs.get("init_u", None)

        if init_u is None:
            init_u = self.rng.uniform(size=(kwargs['u_dimension'],)) * 0.1

        l = {}
        match self.optimization_method:
            case 'jaxopt':
                u, l = self.design_stim_jaxopt(v, kwargs['u_dimension'], kwargs['u_to_s_function'], init_u0=init_u)
            case 'cheat_lowd_vec':
                u = (kwargs['equivalent_projection_matrix'] @ v).flatten(),
            case 'cheat_highd_vec_single_neurons':
                u = numpy.zeros(kwargs['equivalent_projection_matrix'].shape[0])
                u[self.rng.choice(kwargs['equivalent_projection_matrix'].shape[0])] = 1
            case 'cheat_highd_vec_many_neurons':
                u = numpy.zeros(kwargs['equivalent_projection_matrix'].shape[0])
                u[self.rng.choice(kwargs['equivalent_projection_matrix'].shape[0], size=self.max_l0_norm, replace=False)] = 1
            case 'cem':
                u,l = self.design_stim_cem(v,kwargs['u_dimension'], kwargs['u_to_s_function'], init_u0=init_u)
            case 'admm':
                u,l = self.design_stim_admm(v,kwargs['u_dimension'], kwargs['u_to_s_function'], init_u0=init_u)
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
