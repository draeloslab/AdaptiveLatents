import copy
from collections import deque
import functools
import time
from itertools import cycle
import jax

from adaptive_latents import StreamingKalmanFilter, ArrayWithTime, Pipeline, StimRegressor, Bubblewrap, proSVD, CenteringTransformer, VJF, KernelSmoother, mmICA, sjPCA
from adaptive_latents.regressions import BaseKernelRegressor
import tqdm.auto as tqdm
import numpy as np
import pandas as pd
from adaptive_latents.stim_designer import StimDesigner

from learn_s_hat_plots import finalize_log


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
):
    stim_time_rng, other_rng = rng.spawn(2)
    sr = StimRegressor(
        autoreg=autoreg(),
        stim_reg=BaseKernelRegressor(length_scale=0.04, maxlen=stim_reg_maxlen),
        stim_designer=StimDesigner(max_l0_norm=max_l0_norm, rng_seed=other_rng.integers(2 ** 32), should_log=True),
        # stim_designer=StimDesigner(max_l0_norm=max_l0_norm, max_inner_iters=500, max_outer_loop_time_ms=5000, convergence_threshold=10**-2, adam_learning_rate=10**-2, rng_seed=other_rng.integers(2 ** 32), should_log=True),
        log_level=2,
        check_dt=True,
        attempt_correction=attempt_correction,
        heed_stimuli=heed_stimuli,
        stim_delay=regressor_stim_delay,
    )

    static_S_seed = other_rng.integers(2 ** 32)

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
    high_d_without_stim = []
    high_d_stims = []
    if stim_timing_method == 'regular':
        last_stim_t = initial_nostim_period
        if regular_stim_iter is not None:
            assert stim_rate is None
            regular_stim_iter = copy.deepcopy(regular_stim_iter)
            stim_rate = next(regular_stim_iter)
    for data in Pipeline().streaming_run_on(input_array):
        log_stim_reg_after_stim = False

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

        decided_stims.append(ArrayWithTime(stim_decision, data.t))

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
                if design_method == 'optimized learned u_to_s':
                    if sr.stim_reg.n_observed > n_identity_prior:
                        f = sr.stim_reg.make_jax_pred_f()
                        pred = sr.autoreg.predict(n_steps=0)
                        def u_to_s_function(u):
                            return stim_magnitude * f(jax.numpy.hstack((pred, u)))
                    else:
                        def u_to_s_function(u):
                            return stim_magnitude * equivalent_projection_matrix.T @ u
                elif design_method == 'optimized identity u_to_s':
                    def u_to_s_function(u):
                        return stim_magnitude * equivalent_projection_matrix.T @ u
                else:
                    raise ValueError()

                designed_stim = sr.stim_designer.design_stim(desired_stim, u_to_s_function=u_to_s_function, u_dimension=equivalent_projection_matrix.shape[0])

                log_stim_reg_after_stim = True
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

            log_stim_reg_after_stim = True

            if 'optimized' not in design_method:
                sr.stim_designer.log.append({'v': desired_stim, 'u': designed_stim, 's': desired_stim * np.nan})

            sr.stim_designer.log[-1]['stim_reg'] = copy.deepcopy(sr.stim_reg)
            sr.stim_designer.log[-1]['time_of_stim'] = data.t
            sr.stim_designer.log[-1]['equiv_proj_mat'] = equivalent_projection_matrix

            instantaneous_stim = designed_stim * stim_magnitude
        else:
            instantaneous_stim = np.zeros(input_array.shape[1])

        # latent_position = centerer.transform(data, stream= 'X')
        # latent_position = pro.transform(latent_position, stream='X')

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

        # data_no_stim = data
        # nostim_centerer = copy.deepcopy(centerer)
        # nostim_smoother = copy.deepcopy(smoother)
        # data_no_stim = nostim_centerer.partial_fit_transform(data_no_stim, stream= 'X')
        # data_no_stim = nostim_smoother.partial_fit_transform(data_no_stim, stream= 'X')
        # high_d_without_stim.append(nostim_centerer.inverse_transform(data_no_stim))

        to_add = to_add + delayed_stim
        high_d_without_stim.append(data)
        high_d_stims.append(ArrayWithTime(to_add, data.t))
        data = data + to_add
        to_add = decay_rate * to_add
        data = centerer.partial_fit_transform(data, stream= 'X')
        data = smoother.partial_fit_transform(data, stream= 'X')
        data = pro.partial_fit_transform(data, stream='X')
        if last_dim_red_object is not None:
            data = last_dim_red_object.partial_fit_transform(data, stream='X')
        latents.append(data)


        sr.partial_fit_transform(ArrayWithTime(transformed_instantaneous_stim, data.t), stream= 'stim')
        data = sr.partial_fit_transform(data, stream= 'X')

        if log_stim_reg_after_stim and heed_stimuli:
            overflow, to_grab_idx = divmod(sr.stim_reg.n_observed, sr.stim_reg.history.shape[0])
            newest_row = sr.stim_reg.history[to_grab_idx-1]
            assert overflow or np.isnan(sr.stim_reg.history[to_grab_idx]).all()
            sr.stim_designer.log[-1]['observed_s_hat'] = newest_row[-sr.stim_reg.output_d:]
            sr.stim_designer.log[-1]['observed_reg_input'] = newest_row[:-sr.stim_reg.output_d]

        if data.t > exit_time:
            break

    sr.log['high_d_without_stim'] = ArrayWithTime.from_list(high_d_without_stim, squeeze_type='to_2d', drop_early_nans=True)
    sr.log['high_d_stims'] = ArrayWithTime.from_list(high_d_stims, squeeze_type='to_2d', drop_early_nans=True)
    sr.log['latents'] = ArrayWithTime.from_list(latents, squeeze_type='to_2d', drop_early_nans=True)
    finalize_log(sr, ArrayWithTime.from_list(decided_stims, squeeze_type='to_2d'))

    # import matplotlib.pyplot as plt
    # latents = ArrayWithTime.from_list(latents, drop_early_nans=True, squeeze_type='to_2d')
    # e = sr.log['pred_error']
    # l, e = ArrayWithTime.align_indices(latents, e)
    # fig, axs = plt.subplots(nrows=2, sharex=True)
    #
    # axs[0].plot(l.t, l[:,0], '.-')
    # axs[0].plot(l.t, l[:,0] + e[:,0], '.-')
    # axs[0].set_title(f'{stim_time_delay=}, {regressor_stim_delay=}')
    # for s in sr.log['stim_intended_samples']:
    #     axs[0].axvline(s.t, color='k')
    #
    # axs[1].plot(e.t, e[:,0], '.-')
    # axs[1].set_title(np.nanmean(e.slice(slice(input_array.shape[0]//2, None))[:,0]**2))
    # for s in sr.log['stim_intended_samples']:
    #     axs[0].axvline(s.t, color='k')
    # plt.show(block=True)

    return sr

def make_srs(data, rng, comparison_preset=None, n_runs=1, show_tqdm=False, overrides=None):
    if overrides is None:
        overrides = {}

    to_run = get_presets(comparison_preset)

    srs = {}
    with tqdm.tqdm(total=len(to_run) * n_runs, disable=not show_tqdm) as pbar:
        for key, val in to_run.items():
            val = val | overrides
            sub_rng = copy.deepcopy(rng)
            srs[key] = []
            for _ in range(n_runs):
                srs[key].append(make_sr(input_array=data, rng=sub_rng, **val))
                pbar.update(1)

    return srs


def get_presets(comparison_preset):
    match comparison_preset:
        case 'pred methods':
            to_run = {
                'kf': dict(autoreg=StreamingKalmanFilter),
                'bw':dict(autoreg=Bubblewrap),
                'vjf':dict(autoreg=VJF)
            }
        case 'optim_col_vs_rand':
            design_method = 'optimized identity u_to_s'
            stim_rate=1/2
            exit_time=130
            to_run = {
                'first column of Q': dict(design_method=design_method, stim_direction_type='first', stim_rate=stim_rate, exit_time=exit_time,),
                'random columns of Q': dict(design_method=design_method, stim_direction_type='col',stim_rate=stim_rate, exit_time=exit_time,),
                'random unit vector': dict(design_method=design_method, stim_direction_type='random', stim_rate=stim_rate, exit_time=exit_time,),
            }

        case 'optim_col_vs_rand_with_high_d_rand':
            common = dict(stim_direction_type='first', stim_rate=1/2, stim_magnitude=10, exit_time=130)
            to_run = {
                'normal': common | dict( true_S='identity', design_method='optimized identity u_to_s',),
                'shuffled': common | dict(true_S='high_d_permuted',design_method='optimized identity u_to_s'),
                'many': common | dict(true_S='identity',design_method='many neurons'),
                'single': common | dict(true_S='identity', design_method='single neurons'),
                # 'near_zero': common | dict(true_S='identity', design_method='optimized identity u_to_s', stim_magnitude=0.001),
            }

        case 'optim_open_vs_closed':
            stim_rate = 1/2
            exit_time = np.inf
            prosvd_k = 10
            to_run = {
                'open id': dict(design_method='optimized identity u_to_s', true_S='identity', stim_direction_type='first', stim_rate=stim_rate, exit_time=exit_time, prosvd_k=prosvd_k,),
                'closed id': dict(design_method='optimized learned u_to_s', true_S='identity', stim_direction_type='first',stim_rate=stim_rate, exit_time=exit_time, prosvd_k=prosvd_k,),
                'open flip': dict(design_method='optimized identity u_to_s', true_S='flip', stim_direction_type='first', stim_rate=stim_rate, exit_time=exit_time, prosvd_k=prosvd_k, ),
                'closed flip': dict(design_method='optimized learned u_to_s', true_S='flip', stim_direction_type='first', stim_rate=stim_rate, exit_time=exit_time, prosvd_k=prosvd_k, ),
            }
        case 'optim_open_vs_closed_toy':
            stim_rate = 3
            exit_time = np.inf
            prosvd_k = 2
            to_run = {
                'open id': dict(design_method='optimized identity u_to_s', true_S='identity', stim_direction_type='first', stim_rate=stim_rate, exit_time=exit_time, prosvd_k=prosvd_k,),
                'closed id': dict(design_method='optimized learned u_to_s', true_S='identity', stim_direction_type='first',stim_rate=stim_rate, exit_time=exit_time, prosvd_k=prosvd_k,),
                'open flip': dict(design_method='optimized identity u_to_s', true_S='flip', stim_direction_type='first', stim_rate=stim_rate, exit_time=exit_time, prosvd_k=prosvd_k, ),
                'closed flip': dict(design_method='optimized learned u_to_s', true_S='flip', stim_direction_type='first', stim_rate=stim_rate, exit_time=exit_time, prosvd_k=prosvd_k, ),
            }
        case 'delay-table':
            to_run = {}
            common = dict(stim_magnitude=10, prosvd_k=8, exit_time=30, initial_nostim_period=5, design_method='direct cheating')

            # for LDS
            # common |= dict(prosvd_k=4, exit_time=np.inf, initial_nostim_period=10, stim_rate=1 / 20)

            for i in range(4):
                for j in range(4):
                    # for LDS:
                    # to_run[f'({i}, {j})'] = dict(stim_time_delay=i, regressor_stim_delay=j, stim_magnitude=10, prosvd_k=4, exit_time=np.inf, initial_nostim_period=10, design_method='direct cheating', stim_rate=1/20)
                    # for ODoherty
                    to_run[f'({i}, {j})'] = common | dict(stim_time_delay=i, regressor_stim_delay=j)

        case 'default':
            stim_magnitude = 10
            design_method = 'optimized identity u_to_s'
            exit_time = np.inf
            stim_rate = None
            smoothing_tau = 1
            centerer_init_size = 8 * 25
            initial_nostim_period = 30
            regular_stim_iter = cycle([1 / 10, 1 / 3])
            stim_timing_method = 'regular'
            autoreg=functools.partial(StreamingKalmanFilter, steps_between_refits=5)

            to_run = {
                'learning from stim': dict(attempt_correction=True, heed_stimuli=True, exit_time=exit_time, stim_magnitude=stim_magnitude, design_method=design_method, stim_rate=stim_rate, smoothing_tau=smoothing_tau, centerer_init_size=centerer_init_size, initial_nostim_period=initial_nostim_period,regular_stim_iter=regular_stim_iter,stim_timing_method=stim_timing_method, autoreg=autoreg,),
                'ignoring stim samples':dict(attempt_correction=False, heed_stimuli=True, exit_time=exit_time,stim_magnitude=stim_magnitude, design_method=design_method,stim_rate=stim_rate, smoothing_tau=smoothing_tau,centerer_init_size=centerer_init_size,initial_nostim_period=initial_nostim_period,regular_stim_iter=regular_stim_iter,stim_timing_method=stim_timing_method, autoreg=autoreg,),
                'unaware of stim':dict(attempt_correction=False, heed_stimuli=False, exit_time=exit_time, stim_magnitude=stim_magnitude,design_method=design_method,stim_rate=stim_rate, smoothing_tau=smoothing_tau,centerer_init_size=centerer_init_size,initial_nostim_period=initial_nostim_period,regular_stim_iter=regular_stim_iter,stim_timing_method=stim_timing_method, autoreg=autoreg,)
            }
        case 'visualization':
            stim_magnitude = 10
            design_method = 'optimized identity u_to_s'
            exit_time = np.inf
            stim_rate = None
            smoothing_tau = 1
            centerer_init_size = 8 * 25
            initial_nostim_period = 30
            regular_stim_iter = cycle([1 / 10, 1 / 3])
            stim_timing_method = 'regular'

            to_run = {
                'learning from stim': dict(attempt_correction=True, heed_stimuli=True, exit_time=exit_time,
                                           stim_magnitude=stim_magnitude, design_method=design_method, stim_rate=stim_rate,
                                           smoothing_tau=smoothing_tau, centerer_init_size=centerer_init_size,
                                           initial_nostim_period=initial_nostim_period,
                                           regular_stim_iter =regular_stim_iter, stim_timing_method=stim_timing_method),
            }

        case _:
            raise ValueError()

    return to_run




stim_dim_slice = 5
time_slices = ('post-stim', 'non-stim', 'all')
space_slices = ('stim-d', 'non-stim-d', 'all')
def make_slices_tensor(sr):
    error = sr.log['pred_error']
    stim_intended_samples = sr.log['stim_intended_samples']

    outputs = []
    index_one = []
    index_two = []

    for i, time_slice in enumerate(time_slices):
        match time_slice:
            case 'post-stim':
                _, value = ArrayWithTime.align_indices(stim_intended_samples, error)
            case 'non-stim':
                _, value = ArrayWithTime.align_indices(stim_intended_samples, error, complement=True)
            case 'all':
                value = error
            case _:
                raise ValueError()

        for j, space_slice in enumerate(space_slices):
            match space_slice:
                case 'stim-d':
                    value2 = value[:, :stim_dim_slice]
                case 'non-stim-d':
                    value2 = value[:, stim_dim_slice:]
                case 'all':
                    value2 = value
                case _:
                    raise ValueError()

            outputs.append(value2)
            index_one.append(time_slice)
            index_two.append(space_slice)
    outputs = pd.Series(outputs, index=[np.array(index_one), np.array(index_two)])
    return outputs
