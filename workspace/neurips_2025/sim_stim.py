import copy
from collections import deque
import functools
import time
import jax

from adaptive_latents import StreamingKalmanFilter, ArrayWithTime, Pipeline, StimRegressor, Bubblewrap, proSVD, CenteringTransformer, VJF
from adaptive_latents.regressions import BaseKernelRegressor
import tqdm.auto as tqdm
import numpy as np
import pandas as pd
from adaptive_latents.stim_designer import StimDesigner

from learn_s_hat_plots import finalize_log


stim_dim_slice = 5

def make_sr(
        input_array,
        rng,
        autoreg=StreamingKalmanFilter,
        stim_rate=1 / 25,
        decay_rate=.8,
        prosvd_k=10,
        stim_magnitude=10,
        max_l0_norm=30,
        exit_time=30,
        attempt_correction=True,
        heed_stimuli=True,
        stim_time_delay=0,
        regressor_stim_delay=0,
        design_method='optimized',
        initial_nostim_period=5,
):
    dynamics_rng, other_rng = rng.spawn(2)
    sr = StimRegressor(
        autoreg=autoreg(),
        stim_reg=BaseKernelRegressor(length_scale=0.06),
        stim_designer=StimDesigner(max_l0_norm=max_l0_norm, max_inner_iters=50, max_outer_loop_time_ms=5, rng_seed=other_rng.integers(2 ** 32), should_log=True),
        # stim_designer=StimDesigner(max_l0_norm=max_l0_norm, max_inner_iters=500, max_outer_loop_time_ms=5000, convergence_threshold=10**-2, adam_learning_rate=10**-2, rng_seed=other_rng.integers(2 ** 32), should_log=True),
        log_level=2,
        check_dt=True,
        attempt_correction=attempt_correction,
        heed_stimuli=heed_stimuli,
        stim_delay=regressor_stim_delay,
    )

    centerer = CenteringTransformer()
    pro = proSVD(k=prosvd_k)

    stim_delay_queue = deque([0]*stim_time_delay)

    to_add = np.zeros(input_array.shape[1])
    decided_stims = []
    latents = []
    for data in Pipeline().streaming_run_on(input_array):

        stim_decision = data.t > initial_nostim_period and dynamics_rng.random() < stim_rate

        decided_stims.append(ArrayWithTime(stim_decision, data.t))
        if stim_decision and pro.Q is not None:
            if design_method == 'optimized':
                _time_start = time.time()
                if sr.stim_reg.n_observed > 10 and False:
                    f = sr.stim_reg.make_jax_pred_f()
                    def u_to_s_function(u):
                        return f(jax.numpy.hstack((sr.autoreg.predict(n_steps=0), u)))
                    designed_stim = sr.stim_designer.design_stim(pro.Q[:, :stim_dim_slice], u_to_s_function=u_to_s_function)
                else:
                    def u_to_s_function(u):
                        return pro.Q.T @ u

                    desired_stim = np.zeros((pro.Q.shape[1], 1))
                    desired_stim[0] = 1
                    designed_stim = sr.stim_designer.design_stim(desired_stim, u_to_s_function=u_to_s_function, u_dimension=pro.Q.shape[0])

                sr.stim_designer.log[-1]['stim_reg'] = copy.deepcopy(sr.stim_reg)
                sr.stim_designer.log[-1]['pro'] = copy.deepcopy(pro)
                print(f'{(time.time() - _time_start) * 1000:.1f}')
            elif design_method == 'direct cheating':
                designed_stim = pro.Q[:,0]
            else:
                raise NotImplementedError()
            instantaneous_stim = designed_stim * stim_magnitude
        else:
            instantaneous_stim = np.zeros(input_array.shape[1])

        stim_delay_queue.appendleft(instantaneous_stim)
        delayed_stim = stim_delay_queue.pop()

        to_add = to_add + delayed_stim

        data = data + to_add
        to_add = decay_rate * to_add
        data = centerer.partial_fit_transform(data, stream= 'X')
        data = pro.partial_fit_transform(data, stream='X')
        latents.append(data)


        sr.partial_fit_transform(ArrayWithTime(instantaneous_stim, data.t), stream= 'stim')
        data = sr.partial_fit_transform(data, stream= 'X')

        if data.t > exit_time:
            break

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

def make_srs(data, rng, comparison_preset=None, n_runs=1, show_tqdm=False):
    match comparison_preset:
        case 'pred methods':
            to_run = {
                'kf': dict(autoreg=StreamingKalmanFilter),
                'bw':dict(autoreg=Bubblewrap),
                'vjf':dict(autoreg=VJF)
            }
        case 'delay-table':
            to_run = {}
            for i in range(4):
                for j in range(4):
                    # for LDS:
                    to_run[f'({i}, {j})'] = dict(stim_time_delay=i, regressor_stim_delay=j, stim_magnitude=10, prosvd_k=4, exit_time=np.inf, initial_nostim_period=10, design_method='direct cheating')
                    # for ODoherty
                    # to_run[f'({i}, {j})'] = dict(stim_time_delay=i, regressor_stim_delay=j, stim_magnitude=10, prosvd_k=8, exit_time=30, initial_nostim_period=5, design_method='direct cheating')

        case 'default':
            to_run = {
                'learning from stim': dict(attempt_correction=True, heed_stimuli=True),
                'ignoring stim samples':dict(attempt_correction=False, heed_stimuli=True),
                'unaware of stim':dict(attempt_correction=False, heed_stimuli=False)
            }
        case _:
            raise ValueError()

    srs = {}
    with tqdm.tqdm(total=len(to_run) * n_runs, disable=not show_tqdm) as pbar:
        for key, val in to_run.items():
            sub_rng = copy.deepcopy(rng)
            srs[key] = []
            for _ in range(n_runs):
                srs[key].append(make_sr(input_array=data, rng=sub_rng, **val))
                pbar.update(1)

    return srs





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
