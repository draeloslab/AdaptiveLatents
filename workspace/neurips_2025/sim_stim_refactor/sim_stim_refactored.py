import copy
from collections import deque
import functools
from itertools import cycle, chain
import jax
import warnings

from adaptive_latents import StreamingKalmanFilter, ArrayWithTime, Pipeline, StimRegressor, Bubblewrap, proSVD, CenteringTransformer, VJF, KernelSmoother, mmICA, sjPCA
from adaptive_latents.regressions import BaseKernelRegressor
import numpy as np
from adaptive_latents.stim_designer import StimDesigner
from tqdm.auto import tqdm
from contextlib import nullcontext


from adaptive_latents.transformer import StreamingTransformer


class SimulatedStimAdder(StreamingTransformer):
    def __init__(self, *, tau=1, true_S, static_S_seed, decay=.8, stim_time_delay=0, input_streams=None, output_streams=None, log_level=None):
        input_streams = input_streams or {0:'X'}
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)

        self.true_S = true_S
        self.static_S_seed = static_S_seed

        # self.tau = tau
        # delta_t = 1
        # self.alpha = 1 - np.exp(-delta_t/tau)
        self.alpha = decay

        self.to_add = 0

        self.stim_time_delay = stim_time_delay
        self.stim_delay_queue = deque([0] * stim_time_delay)

    def register_stim(self, true_stim_result):
        self.stim_delay_queue.appendleft(true_stim_result)

    def _partial_fit_transform(self, data, stream, return_output_stream):
        if self.input_streams[stream] == 'X':
            self.to_add += self.stim_delay_queue.pop()
            data = data + self.to_add
            self.to_add = self.to_add * self.alpha
        if self.input_streams[stream] == 'stim':
            # TODO: check for regularity?
            self.register_stim(data)

        stream = self.output_streams[stream]
        return (data, stream) if return_output_stream else data

    def true_stim_result(self, instantaneous_stim, equivalent_projection_matrix=None):
        if self.true_S == 'identity':
            transformed_instantaneous_stim = instantaneous_stim
        elif self.true_S == 'flip':
            if equivalent_projection_matrix is not None:
                in_space_comp = equivalent_projection_matrix.T @ instantaneous_stim
                out_of_space_comp = instantaneous_stim - equivalent_projection_matrix @ in_space_comp
                transformed_instantaneous_stim = equivalent_projection_matrix @ in_space_comp[::-1] + out_of_space_comp
            else:
                assert (instantaneous_stim == 0).all()
                transformed_instantaneous_stim = instantaneous_stim
        elif self.true_S == 'high_d_permuted':
            transformed_instantaneous_stim = np.random.default_rng(self.static_S_seed).permuted(instantaneous_stim)
        else:
            raise ValueError(self.true_S)

        return transformed_instantaneous_stim


def calculate_equivalent_projection_matrix(pro, last_dim_red_object):
    equivalent_projection_matrix = pro.Q
    if equivalent_projection_matrix is not None:
        if isinstance(last_dim_red_object, sjPCA):
            try:
                U = last_dim_red_object.get_U()
            except AttributeError: # TODO make this more elegant
                U = None
            if U is not None:
                equivalent_projection_matrix = equivalent_projection_matrix @ U
        elif isinstance(last_dim_red_object, mmICA):
            W = last_dim_red_object.W
            if W is not None:
                equivalent_projection_matrix = equivalent_projection_matrix @ W.T
        elif last_dim_red_object is None:
            pass
        else:
            raise ValueError()
    return equivalent_projection_matrix


def design_stim(stim_designer, sr, stim_magnitude, desired_stim, equivalent_projection_matrix, current_t):
    optimization_method = stim_designer.optimization_method
    u_to_s_model_type = stim_designer.u_to_s_model_type
    if u_to_s_model_type == 'kernel_regressed' and sr.stim_reg.n_observed <= stim_designer.n_identity_initialization:
        u_to_s_model_type = 'identity'


    if optimization_method == 'jaxopt' and u_to_s_model_type == 'kernel_regressed':
        f = sr.stim_reg.make_jax_pred_f()
        pred = sr.autoreg.predict(n_steps=0)
        def u_to_s_function(u):
            return stim_magnitude * f(jax.numpy.hstack((pred, u)))
        designed_stim = stim_designer.design_stim(desired_stim, u_to_s_function=u_to_s_function, u_dimension=equivalent_projection_matrix.shape[0])
    elif optimization_method == 'jaxopt' and u_to_s_model_type == 'identity':
        def u_to_s_function(u):
            return stim_magnitude * equivalent_projection_matrix.T @ u
        designed_stim = stim_designer.design_stim(desired_stim, u_to_s_function=u_to_s_function, u_dimension=equivalent_projection_matrix.shape[0])
    elif optimization_method == 'cheat' and u_to_s_model_type == 'identity':
        designed_stim = stim_designer.design_stim(desired_stim, equivalent_projection_matrix=equivalent_projection_matrix)
    else:
        raise ValueError()

    stim_designer.log[-1]['stim_reg'] = copy.deepcopy(sr.stim_reg)
    stim_designer.log[-1]['time_of_stim'] = current_t
    stim_designer.log[-1]['equiv_proj_mat'] = equivalent_projection_matrix

    return designed_stim


def make_sr(
        input_array,
        rng,
        autoreg=StreamingKalmanFilter,
        stim_rate=1, # TODO: refactor out
        regular_stim_iter=None,  # TODO: refactor out
        isi_generator=None,
        exit_time=60,
        decay_rate=.8,
        prosvd_k=10,
        stim_magnitude=10,
        max_l0_norm=30,
        attempt_correction=True,
        heed_stimuli=True,
        stim_time_delay=0,
        regressor_stim_delay=0,
        design_method='optimized identity u_to_s', # TODO: refactor out
        design_type=None,
        u_to_s_model_type=None,
        true_S='identity',
        stim_timing_method='random',
        n_identity_prior=10,
        stim_direction_type='first',
        initial_nostim_period=5,
        stim_reg_maxlen=500,
        smoothing_tau=None,
        centerer_init_size=0,
        last_dim_red='prosvd',
        show_tqdm=False,
):
    assert (regular_stim_iter is not None) + (stim_rate is not None) + (isi_generator is not None) == 1
    if stim_rate:
        isi_generator = cycle([1/stim_rate])
    elif regular_stim_iter:
        isi_generator = map(lambda x: 1/x, regular_stim_iter)
        assert stim_timing_method == 'regular'
        stim_timing_method = 'isi'
    del regular_stim_iter, stim_rate

    optimization_method, u_to_s_model_type = {
        'optimized learned u_to_s': ('jaxopt', 'kernel_regressed'),
        'optimized identity u_to_s': ('jaxopt', 'identity'),
        'direct cheating': ('cheat', 'identity'),
    }[design_method]
    # single neurons
    # many neurons
    # del design_method

    stim_time_rng, other_rng = rng.spawn(2)


    sr = StimRegressor(
        autoreg=autoreg(),
        stim_reg=BaseKernelRegressor(length_scale=0.04, maxlen=stim_reg_maxlen),
        log_level=2,
        check_dt=True,
        attempt_correction=attempt_correction,
        heed_stimuli=heed_stimuli,
        stim_delay=regressor_stim_delay,
    )
    stim_designer = StimDesigner(
        max_l0_norm=max_l0_norm,
        rng_seed=other_rng.integers(2 ** 32),
        should_log=True,
        initial_nostim_period=initial_nostim_period,
        stim_timing_method=stim_timing_method,
        inter_stim_interval_generator=isi_generator,
        optimization_method=optimization_method, # todo:fix
        u_to_s_model_type=u_to_s_model_type,
        n_identity_initialization=n_identity_prior
    )

    static_S_seed = other_rng.integers(2 ** 32)
    sim_stim_adder = SimulatedStimAdder(
        true_S=true_S,
        static_S_seed=static_S_seed,
        stim_time_delay=stim_time_delay,
        decay=decay_rate
    )

    log = {}


    centerer = CenteringTransformer(init_size=centerer_init_size, nan_when_uninitialized=True)
    if smoothing_tau is not None:
        smoother = KernelSmoother(tau=smoothing_tau/input_array.dt)
    else:
        smoother = Pipeline()

    pro = proSVD(k=prosvd_k)
    if last_dim_red == 'prosvd':
        last_dim_red_object = None
    elif last_dim_red == 'sjpca':
        last_dim_red_object = sjPCA()
    elif last_dim_red == 'mmica':
        last_dim_red_object = mmICA()
    else:
        raise ValueError()

    decided_stims = []
    latents = []
    high_d_without_stim = []
    high_d_stims = []

    pbar = nullcontext()
    if show_tqdm:
        pbar = tqdm(total=min(input_array.t[-1], exit_time))

    with pbar:
        for data in Pipeline().streaming_run_on(input_array):
            log_stim_reg_after_stim = False

            stim_decision = stim_designer.decide_whether_to_stim(data.t, stim_time_rng=stim_time_rng, input_array_dt=input_array.dt)
            decided_stims.append(ArrayWithTime(stim_decision, data.t))

            equivalent_projection_matrix = calculate_equivalent_projection_matrix(pro, last_dim_red_object)
            if stim_decision and equivalent_projection_matrix is not None:
                desired_stim = stim_designer.desired_stim_direction(equivalent_projection_matrix, stim_direction_type, other_rng)
                designed_stim = design_stim(stim_designer, sr, stim_magnitude, desired_stim, equivalent_projection_matrix, current_t=data.t)
                log_stim_reg_after_stim = True
                instantaneous_stim = designed_stim * stim_magnitude
            else:
                instantaneous_stim = np.zeros(input_array.shape[1])

            true_stim_result = sim_stim_adder.true_stim_result(instantaneous_stim, equivalent_projection_matrix)

            sim_stim_adder.partial_fit_transform(true_stim_result, stream='stim')

            high_d_without_stim.append(data)
            pre_stim_data = data
            data = sim_stim_adder.partial_fit_transform(data, stream='X')
            high_d_stims.append(data - pre_stim_data)

            data = centerer.partial_fit_transform(data, stream= 'X')
            data = smoother.partial_fit_transform(data, stream= 'X')
            data = pro.partial_fit_transform(data, stream='X')
            if last_dim_red_object is not None:
                data = last_dim_red_object.partial_fit_transform(data, stream='X')
            latents.append(data)

            sr.partial_fit_transform(ArrayWithTime(true_stim_result, data.t), stream= 'stim')
            data = sr.partial_fit_transform(data, stream= 'X')

            if log_stim_reg_after_stim and heed_stimuli:
                overflow, to_grab_idx = divmod(sr.stim_reg.n_observed, sr.stim_reg.history.shape[0])
                newest_row = sr.stim_reg.history[to_grab_idx-1]
                assert overflow or np.isnan(sr.stim_reg.history[to_grab_idx]).any()
                stim_designer.log[-1]['observed_s_hat'] = newest_row[-sr.stim_reg.output_d:]
                stim_designer.log[-1]['observed_reg_input'] = newest_row[:-sr.stim_reg.output_d]

            if show_tqdm:
                pbar.update(round(float(data.t), 2) - pbar.n)
            if data.t > exit_time:
                break

    log['high_d_stims'] = ArrayWithTime.from_list(high_d_stims, squeeze_type='to_2d', drop_early_nans=True)
    log['high_d_without_stim'] = ArrayWithTime.from_list(high_d_without_stim, squeeze_type='to_2d', drop_early_nans=True)
    log['latents'] = ArrayWithTime.from_list(latents, squeeze_type='to_2d', drop_early_nans=True)
    if (log['high_d_stims'] == 0).all():
        warnings.warn("No stims delivered in sim-stim.")

    stim_intended_samples = ArrayWithTime.from_list(decided_stims, squeeze_type='to_2d')
    log['stim_intended_samples'] = stim_intended_samples.slice((stim_intended_samples > 0).any(axis=1))


    return sr, stim_designer, log
