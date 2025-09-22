import copy
from collections import deque
import functools
import time
from itertools import cycle
import jax

from adaptive_latents import StreamingKalmanFilter, ArrayWithTime, Pipeline, StimRegressor, Bubblewrap, proSVD, CenteringTransformer, VJF, KernelSmoother, mmICA, sjPCA
import tqdm.auto as tqdm
import numpy as np
import pandas as pd
from adaptive_latents.stim_designer import StimDesigner, OptimizationMethod
from adaptive_latents.sim_stim import make_sr as new_make_sr



def make_sr(*args, **kwargs):
    sr, stim_designer, log = new_make_sr(*args, **kwargs)
    sr.log.update(log)
    sr.stim_designer = stim_designer

    # sr.log['stim_intended_samples'] = ArrayWithTime.from_list(sr.log['stim_intended_samples'])
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
    default_common = dict(stim_magnitude=10, optimization_method=OptimizationMethod.JAXOPT, u_to_s_model_type='identity', exit_time=np.inf, stim_rate=None, smoothing_tau=1, centerer_init_size=8 * 25, initial_nostim_period=30, regular_stim_iter=cycle([1 / 10, 1 / 3]), stim_timing_method='regular', autoreg=functools.partial(StreamingKalmanFilter, steps_between_refits=5), )

    match comparison_preset:
        case 'pred methods':
            to_run = {
                'kf': dict(autoreg=StreamingKalmanFilter),
                'bw':dict(autoreg=Bubblewrap),
                'vjf':dict(autoreg=VJF)
            }
        case 'optim_col_vs_rand':
            common = dict(optimization_method=OptimizationMethod.JAXOPT, u_to_s_model_type='identity', stim_rate=1/2, exit_time=130)
            to_run = {
                'first column of Q': common | dict(stim_direction_type='first'),
                'random columns of Q': common | dict(stim_direction_type='col'),
                'random unit vector': common | dict(stim_direction_type='random'),
            }

        case 'optim_col_vs_rand_with_high_d_rand':
            common = dict(stim_rate=1/2, stim_magnitude=10, exit_time=130)
            to_run = {}
            stim_direction_types = ('first', 'ones', 'random+', 'col', 'random', '-ones')
            for stim_direction_type in stim_direction_types:
                inner_common = common | dict(stim_direction_type=stim_direction_type)
                to_run.update({
                    f'normal {stim_direction_type}': inner_common | dict(true_S='identity', optimization_method=OptimizationMethod.JAXOPT, u_to_s_model_type='identity',),
                    f'shuffled {stim_direction_type}': inner_common | dict(true_S='high_d_permuted', optimization_method=OptimizationMethod.JAXOPT, u_to_s_model_type='identity'),
                    f'many {stim_direction_type}': inner_common | dict(true_S='identity', optimization_method=OptimizationMethod.CHEAT_HIGHD_VEC_MANY_NEURONS, u_to_s_model_type=None),
                    f'single {stim_direction_type}': inner_common | dict(true_S='identity', optimization_method=OptimizationMethod.CHEAT_HIGHD_VEC_SINGLE_NEURONS, u_to_s_model_type=None),
                })

        case 'optim_open_vs_closed':
            common = dict(stim_rate = 1/2, exit_time = np.inf, prosvd_k = 10, optimization_method=OptimizationMethod.JAXOPT, stim_direction_type='first',)
            to_run = {
                'open id': common | dict(u_to_s_model_type='identity', true_S='identity'),
                'closed id': common | dict(u_to_s_model_type='kernel_regressed', true_S='identity'),
                'open flip': common | dict(u_to_s_model_type='identity', true_S='flip'),
                'closed flip': common | dict(u_to_s_model_type='kernel_regressed', true_S='flip',),
            }
        case 'optim_open_vs_closed_toy':
            common = dict( stim_rate = 3, exit_time = np.inf, prosvd_k = 2, optimization_method=OptimizationMethod.JAXOPT,stim_direction_type='first',)
            to_run = {
                'open id': common | dict( u_to_s_model_type='identity', true_S='identity'),
                'closed id': common | dict(u_to_s_model_type='kernel_regressed', true_S='identity'),
                'open flip': common | dict(u_to_s_model_type='identity', true_S='flip'),
                'closed flip': common | dict(u_to_s_model_type='kernel_regressed', true_S='flip'),
            }
        case 'delay-table':
            to_run = {}
            common = dict(stim_magnitude=10, prosvd_k=8, exit_time=30, initial_nostim_period=5, optimization_method=OptimizationMethod.CHEAT_LOWD_VEC, u_to_s_model_type='identity')

            # for LDS
            # common |= dict(prosvd_k=4, exit_time=np.inf, initial_nostim_period=10, stim_rate=1 / 20)

            for i in range(4):
                for j in range(4):
                    # for LDS:
                    # to_run[f'({i}, {j})'] = dict(stim_time_delay=i, regressor_stim_delay=j, stim_magnitude=10, prosvd_k=4, exit_time=np.inf, initial_nostim_period=10, optimization_method='cheat_lowd_vec', u_to_s_model_type='identity', stim_rate=1/20)
                    # for ODoherty
                    to_run[f'({i}, {j})'] = common | dict(stim_time_delay=i, regressor_stim_delay=j)

        case 'default':
            common = default_common
            to_run = {
                'learning from stim': common | dict(attempt_correction=True, heed_stimuli=True),
                'ignoring stim samples': common | dict(attempt_correction=False, heed_stimuli=True),
                'unaware of stim': common | dict(attempt_correction=False, heed_stimuli=False),
            }
        case 'visualization':
            common = default_common
            del common[autoreg]

            to_run = {
                'learning from stim': common | dict(attempt_correction=True, heed_stimuli=True),
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
