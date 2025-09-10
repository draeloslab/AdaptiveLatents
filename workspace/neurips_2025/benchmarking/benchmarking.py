import copy
from collections import deque
import jax

from adaptive_latents import StreamingKalmanFilter, ArrayWithTime, Pipeline, StimRegressor, Bubblewrap, proSVD, CenteringTransformer, VJF, KernelSmoother, mmICA, sjPCA, datasets
from adaptive_latents.regressions import BaseMultiKernelRegressor
import numpy as np
from adaptive_latents.stim_designer import StimDesigner
import functools

from types import SimpleNamespace
import time


def make_sr(
        input_array,
        rng,
        autoreg=StreamingKalmanFilter,
        stim_rate=1,
        exit_time=60,
        decay_rate=.8,
        prosvd_k=10,
        stim_magnitude=10,
        max_l0_norm=30,
        attempt_correction=True,
        heed_stimuli=True,
        stim_time_delay=0,
        regressor_stim_delay=0,
        design_method='optimized identity u_to_s',
        true_S='identity',
        stim_timing_method='random',
        n_identity_prior=10,
        stim_direction_type='first',
        initial_nostim_period=5,
        stim_reg_maxlen=500,
        smoothing_tau=None,
        centerer_init_size=0,
        regular_stim_iter=None,
        last_dim_red='prosvd',
        lam_1=0.001,
):
    log = SimpleNamespace()
    log.init_time = 0
    log.loop_time = 0
    log.stim_decide = []
    log.stim_design = []
    log.dimension_reduction = []
    log.prediction = []
    log.per_loop = []
    log.stim_reg_updated = []




    log.init_time = time.time()

    stim_time_rng, other_rng = rng.spawn(2)
    sr = StimRegressor(
        autoreg=autoreg(log_level=0),
        stim_reg=BaseMultiKernelRegressor(maxlen=stim_reg_maxlen),
        stim_designer=StimDesigner(max_l0_norm=max_l0_norm, lam_1=lam_1, rng_seed=other_rng.integers(2 ** 32), should_log=False),
        log_level=0,
        check_dt=True,
        attempt_correction=attempt_correction,
        heed_stimuli=heed_stimuli,
        stim_delay=regressor_stim_delay,
    )

    static_S_seed = other_rng.integers(2 ** 32)

    centerer = CenteringTransformer(init_size=centerer_init_size, nan_when_uninitialized=True, log_level=0)
    if smoothing_tau is not None:
        smoother = KernelSmoother(tau=smoothing_tau/input_array.dt)
    else:
        smoother = Pipeline()

    pro = proSVD(k=prosvd_k, log_level=0)
    if last_dim_red == 'prosvd':
        last_dim_red_object = None
    elif last_dim_red == 'sjpca':
        last_dim_red_object = sjPCA(log_level=0)
    elif last_dim_red == 'mmica':
        last_dim_red_object = mmICA(log_level=0)
    else:
        raise ValueError()

    stim_delay_queue = deque([0]*stim_time_delay)

    to_add = np.zeros(input_array.shape[1])
    if stim_timing_method == 'regular':
        last_stim_t = initial_nostim_period
        if regular_stim_iter is not None:
            assert stim_rate is None
            regular_stim_iter = copy.deepcopy(regular_stim_iter)
            stim_rate = next(regular_stim_iter)

    log.init_time = time.time() - log.init_time

    log.loop_time = time.time()
    for data in Pipeline().streaming_run_on(input_array):
        log.per_loop.append(time.time())

        log.stim_decide.append(time.time())
        if stim_timing_method == 'random':
            stim_decision = data.t > initial_nostim_period and stim_time_rng.random() < stim_rate * input_array.dt
        elif stim_timing_method == 'regular':
            stim_decision = False
            if data.t > initial_nostim_period and data.t - last_stim_t > 1/stim_rate:
                stim_decision = True
                last_stim_t = data.t
                if regular_stim_iter is not None:
                    stim_rate = next(regular_stim_iter)
        else:
            raise ValueError()
        log.stim_decide[-1] = time.time() - log.stim_decide[-1]

        equivalent_projection_matrix = pro.Q
        if equivalent_projection_matrix is not None:
            if last_dim_red == 'sjpca':
                try:
                    U = last_dim_red_object.get_U()
                except AttributeError: # TODO make this more elegant
                    U = None
                if U is not None:
                    equivalent_projection_matrix = equivalent_projection_matrix @ U
            if last_dim_red == 'mmica':
                W = last_dim_red_object.W
                if W is not None:
                    equivalent_projection_matrix = equivalent_projection_matrix @ W.T


        log.stim_design.append(time.time())
        if stim_decision and equivalent_projection_matrix is not None:
            if stim_direction_type == 'first':
                desired_stim = np.zeros((equivalent_projection_matrix.shape[1], 1))
                desired_stim[0] = 1
            elif stim_direction_type == 'first2':
                desired_stim = np.zeros((equivalent_projection_matrix.shape[1], 2))
                desired_stim[0] = 1
                desired_stim[1] = 1
            elif stim_direction_type == 'col':
                desired_stim = np.zeros((equivalent_projection_matrix.shape[1], 1))
                desired_stim[other_rng.choice(equivalent_projection_matrix.shape[1]), 0] = 1
            elif stim_direction_type == 'random':
                desired_stim = other_rng.normal(size=(equivalent_projection_matrix.shape[1], 1))
                desired_stim = desired_stim / np.linalg.norm(desired_stim)
            else:
                raise ValueError()

            if 'optimized' in design_method:
                with jax.default_device('cpu'):
                    if design_method == 'optimized learned u_to_s':
                        if sr.stim_reg.n_observed > n_identity_prior:
                            f = sr.stim_reg.make_jax_pred_f()
                            pred = sr.autoreg.predict(n_steps=0)
                            current_t = data.t
                            def u_to_s_function(u):
                                return stim_magnitude * f([pred, u, current_t])
                        else:
                            def u_to_s_function(u):
                                return stim_magnitude * equivalent_projection_matrix.T @ u
                    elif design_method == 'optimized identity u_to_s':
                        def u_to_s_function(u):
                            return stim_magnitude * equivalent_projection_matrix.T @ u
                    else:
                        raise ValueError()

                    designed_stim = sr.stim_designer.design_stim(desired_stim, u_to_s_function=u_to_s_function, u_dimension=equivalent_projection_matrix.shape[0])

            elif design_method == 'direct cheating':
                designed_stim = (equivalent_projection_matrix @ desired_stim).flatten()
            elif design_method == 'single neurons':
                designed_stim = np.zeros(equivalent_projection_matrix.shape[0])
                designed_stim[other_rng.choice(equivalent_projection_matrix.shape[0])] = 1
            elif design_method == 'many neurons':
                designed_stim = np.zeros(equivalent_projection_matrix.shape[0])
                designed_stim[other_rng.choice(equivalent_projection_matrix.shape[0], size=sr.stim_designer.max_l0_norm, replace=False)] = 1
            else:
                raise NotImplementedError()

            instantaneous_stim = designed_stim * stim_magnitude
        else:
            instantaneous_stim = np.zeros(input_array.shape[1])

        log.stim_design[-1] = time.time() - log.stim_design[-1]

        if true_S == 'identity':
            transformed_instantaneous_stim = instantaneous_stim
        elif true_S == 'flip':
            if equivalent_projection_matrix is not None:
                in_space_comp = equivalent_projection_matrix.T @ instantaneous_stim
                out_of_space_comp = instantaneous_stim - equivalent_projection_matrix @ in_space_comp
                transformed_instantaneous_stim = equivalent_projection_matrix @ in_space_comp[::-1] + out_of_space_comp
            else:
                assert (instantaneous_stim == 0).all()
                transformed_instantaneous_stim = instantaneous_stim
        elif true_S == 'high_d_permuted':
            transformed_instantaneous_stim = np.random.default_rng(static_S_seed).permuted(instantaneous_stim)
        else:
            raise ValueError(true_S)

        stim_delay_queue.appendleft(transformed_instantaneous_stim)
        delayed_stim = stim_delay_queue.pop()


        to_add = to_add + delayed_stim
        data = data + to_add
        to_add = decay_rate * to_add

        log.dimension_reduction.append(time.time())
        data = centerer.partial_fit_transform(data, stream= 'X')
        data = smoother.partial_fit_transform(data, stream= 'X')
        data = pro.partial_fit_transform(data, stream='X')
        if last_dim_red_object is not None:
            data = last_dim_red_object.partial_fit_transform(data, stream='X')
        log.dimension_reduction[-1] = time.time() - log.dimension_reduction[-1]


        old_n = sr.stim_reg.n_observed
        log.prediction.append(time.time())
        sr.partial_fit_transform(ArrayWithTime(transformed_instantaneous_stim, data.t), stream= 'stim')
        data = sr.partial_fit_transform(data, stream= 'X')
        log.prediction[-1] = time.time() - log.prediction[-1]
        log.stim_reg_updated.append(old_n != sr.stim_reg.n_observed)

        log.per_loop[-1] = time.time() - log.per_loop[-1]
        if data.t > exit_time:
            break

    log.loop_time = time.time() - log.loop_time

    return log

if __name__ == '__main__':

    rng = np.random.default_rng(0) # sfc64?

    neural_data = datasets.Odoherty21Dataset().neural_data

    log = make_sr(
        neural_data,
        rng,
        # autoreg=functools.partial(Bubblewrap, num=1000, log_level=0),
        exit_time=np.inf,
        design_method = 'optimized learned u_to_s',
        stim_timing_method='regular',
        last_dim_red='sjpca',
        stim_reg_maxlen=100,
        lam_1=1e-6,
    )
    print(f"{log.init_time = }")
    print(f"{log.loop_time = }")

    import pandas as pd

    df = pd.DataFrame({k:v for k,v in log.__dict__.items() if isinstance(v, list)})
    df.to_csv(f'benchmark-{time.strftime("%Y%m%d%H%M%S")}.csv')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(df * 1000)
    ax.set_xlabel('iteration step')
    ax.set_ylabel('time (ms)')

    plt.show()
