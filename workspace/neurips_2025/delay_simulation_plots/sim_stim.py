import copy
from collections import deque
import functools
from itertools import cycle, chain
import jax

from adaptive_latents import StreamingKalmanFilter, ArrayWithTime, Pipeline, StimRegressor, Bubblewrap, proSVD, CenteringTransformer, VJF, KernelSmoother, mmICA, sjPCA
from adaptive_latents.regressions import BaseKernelRegressor
import numpy as np
from adaptive_latents.stim_designer import StimDesigner
from tqdm.auto import tqdm
from contextlib import nullcontext


from adaptive_latents.transformer import StreamingTransformer
class SimulatedStimAdder(StreamingTransformer):
    def __init__(self, *, tau=1, delay=0, u_to_s_callback=None, input_streams=None, output_streams=None, log_level=None):
        input_streams = input_streams or {0:'X'}
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)
        self.tau = tau
        delta_t = 1 # todo: make time-aware
        self.alpha = 1 - np.exp(-delta_t/tau)
        self.to_add = 0
        if u_to_s_callback is None:
            u_to_s_callback = lambda x: x
        self.u_to_s_callback = u_to_s_callback

        assert delay == 0
        self.delay = delay

    def register_stim(self, u):
        self.to_add = self.to_add + self.u_to_s_callback(u)

    def _partial_fit_transform(self, data, stream, return_output_stream):
        if self.input_streams[stream] == 'X':
            data = data + self.to_add
            self.to_add = self.to_add * self.alpha
        stream = self.output_streams[stream]
        return (data, stream) if return_output_stream else data

    def get_params(self, deep=True):
        return dict(tau=self.tau, u_to_s_callback=self.u_to_s_callback, delay=self.delay) | super().get_params()


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


stim_dim_slice = 5

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
        # design_method = 'direct cheating',
        stim_timing_method='random',
        n_identity_prior=10,
        stim_direction_type='first',
        initial_nostim_period=5,
        stim_reg_maxlen=500,
        smoothing_tau=None,
        centerer_init_size=0,
        regular_stim_iter=None,
        last_dim_red='prosvd',
        show_tqdm=False,
):
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
        # inter_stim_interval_generator=chain([30], cycle([1,2]))
        timing_mode='extreme'
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

    stim_delay_queue = deque([0]*stim_time_delay)

    to_add = np.zeros(input_array.shape[1])
    decided_stims = []
    latents = []
    high_d_with_stim = []

    pbar = nullcontext()
    if show_tqdm:
        pbar = tqdm(total=min(input_array.t[-1], exit_time))

    with pbar:
        for data in Pipeline().streaming_run_on(input_array):
            log_stim_reg_after_stim = False

            stim_decision = stim_designer.decide_whether_to_stim(data.t, objective_value=latents[-1][0][0] if len(latents) > 0 else float('inf'))
            decided_stims.append(ArrayWithTime(stim_decision, data.t))

            equivalent_projection_matrix = calculate_equivalent_projection_matrix(pro, last_dim_red_object)



            if stim_decision and equivalent_projection_matrix is not None:
                desired_stim = stim_designer.desired_stim_direction(equivalent_projection_matrix, stim_direction_type, other_rng)


                if 'optimized' in design_method:
                    if design_method == 'optimized learned u_to_s':
                        if sr.stim_reg.n_observed > n_identity_prior:
                            f = sr.stim_reg.make_jax_pred_f()
                            def u_to_s_function(u):
                                return stim_magnitude * f(jax.numpy.hstack((sr.autoreg.predict(n_steps=0), u)))
                        else:
                            def u_to_s_function(u):
                                return stim_magnitude * equivalent_projection_matrix.T @ u
                    elif design_method == 'optimized identity u_to_s':
                        def u_to_s_function(u):
                            return stim_magnitude * equivalent_projection_matrix.T @ u
                    else:
                        raise ValueError()

                    designed_stim = stim_designer.design_stim(desired_stim, u_to_s_function=u_to_s_function, u_dimension=equivalent_projection_matrix.shape[0])

                    log_stim_reg_after_stim = True
                elif design_method == 'direct cheating':
                    designed_stim = (equivalent_projection_matrix @ desired_stim).flatten()
                else:
                    raise NotImplementedError()

                if design_method == 'direct cheating':
                    stim_designer.log.append({})

                stim_designer.log[-1]['stim_reg'] = copy.deepcopy(sr.stim_reg)

                instantaneous_stim = designed_stim * stim_magnitude
            else:
                instantaneous_stim = np.zeros(input_array.shape[1])

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
            else:
                raise ValueError(true_S)

            stim_delay_queue.appendleft(transformed_instantaneous_stim)
            delayed_stim = stim_delay_queue.pop()


            to_add = to_add + delayed_stim
            data = data + to_add
            to_add = decay_rate * to_add
            data = centerer.partial_fit_transform(data, stream= 'X')
            data = smoother.partial_fit_transform(data, stream= 'X')
            high_d_with_stim.append(centerer.inverse_transform(data))
            data = pro.partial_fit_transform(data, stream='X')
            if last_dim_red_object is not None:
                data = last_dim_red_object.partial_fit_transform(data, stream='X')
            latents.append(data)


            sr.partial_fit_transform(ArrayWithTime(instantaneous_stim, data.t), stream= 'stim')
            data = sr.partial_fit_transform(data, stream= 'X')

            if log_stim_reg_after_stim and heed_stimuli and instantaneous_stim.any():
                newest_row = sr.stim_reg.history[sr.stim_reg.n_observed-1]
                assert np.isnan(sr.stim_reg.history[sr.stim_reg.n_observed]).all()
                stim_designer.log[-1]['observed_s_hat'] = newest_row[-sr.stim_reg.output_d:]
                stim_designer.log[-1]['observed_reg_inpt'] = newest_row[:-sr.stim_reg.output_d]

            if show_tqdm:
                pbar.update(round(float(data.t), 2) - pbar.n)
            if data.t > exit_time:
                break

    log['high_d_with_stim'] = ArrayWithTime.from_list(high_d_with_stim, squeeze_type='to_2d', drop_early_nans=True)
    log['latents'] = ArrayWithTime.from_list(latents, squeeze_type='to_2d', drop_early_nans=True)

    stim_intended_samples = ArrayWithTime.from_list(decided_stims, squeeze_type='to_2d')
    log['stim_intended_samples'] = stim_intended_samples.slice((stim_intended_samples > 0).any(axis=1))


    return sr, stim_designer, log