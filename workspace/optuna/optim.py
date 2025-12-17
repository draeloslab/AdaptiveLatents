import logging
import sys
import time

import optuna


import numpy as np
from adaptive_latents import (
    StreamingKalmanFilter, StimRegressor, CenteringEstimator, proSVD, Pipeline, ArrayWithTime, datasets, Bubblewrap, KernelSmoother,
)
from adaptive_latents.regressions import BaseKernelRegressor
from adaptive_latents.stim_designer import StimDesigner
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import functools
import adaptive_latents.plotting_functions


def objective(trial):
    rng = np.random.default_rng()
    d = datasets.Naumann24uDataset(1)

    target_neurons = np.unique(d.opto_stimulations.target_neuron)

    d.neural_data[np.isnan(d.neural_data)] = 0

    def target_neuron_to_vector(tn):
        v = target_neurons * 0
        v[target_neurons == tn] = 1
        return v

    d.opto_stimulations['stim_vector'] = d.opto_stimulations['target_neuron'].apply(target_neuron_to_vector)

    def make_sr(input_array, rng, stim_dict, autoreg=StreamingKalmanFilter, stim_rate=1 / 25, decay_rate=.9, prosvd_k=10, stim_magnitude=10, max_l0_norm=30, exit_time=np.inf, attempt_correction=True, heed_stimuli=True, stim_delay=0):
        sr = StimRegressor(
            autoreg=autoreg(),
            stim_designer=StimDesigner(max_l0_norm=max_l0_norm),
            stim_reg=BaseKernelRegressor(length_scale=trial.suggest_float('length_scale', 0.011253/4, 0.011253*4, log=True), maxlen=20),
            log_level=2,
            check_dt=True,
            attempt_correction=attempt_correction,
            heed_stimuli=heed_stimuli,
            stim_delay=stim_delay,
        )

        centerer = CenteringEstimator()
        pro = proSVD(k=prosvd_k)
        smoother = KernelSmoother(tau=trial.suggest_float('tau', 1.19, 1.19))
        latents = []

        for data in Pipeline().streaming_run_on(input_array):

            if data.t in stim_dict:
                instantaneous_stim = stim_dict[data.t]
            else:
                instantaneous_stim = list(stim_dict.values())[0] * 0
            instantaneous_stim = ArrayWithTime(instantaneous_stim[None,:], data.t)


            data = centerer.step(data, stream='X')
            data = smoother.step(data, stream='X')
            data = pro.step(data, stream='X')

            latents.append(data)

            sr.step(instantaneous_stim, stream='stim')
            sr.step(data, stream='X')

            if data.t > d.end_of_visual_period_time:
                centerer.freeze()
                pro.freeze()

            if data.t > exit_time:
                break
        sr.log['latents'] = ArrayWithTime.from_list(latents, squeeze_type='to_2d')

        return sr

    offset = 8

    stims = ArrayWithTime(d.opto_stimulations['stim_vector'], d.opto_stimulations.time)
    stim_dict = {k: v for k, v in zip(stims.t, stims)}
    try:
        sr = make_sr(input_array=d.neural_data, stim_dict=stim_dict, rng=rng,
                     autoreg=functools.partial(
                         Bubblewrap,
                         num= trial.suggest_int('num', 200, 200),
                         sigma_orig_adjustment=trial.suggest_float('sigma_orig_adjustment', 0, 1000),
                         dead_nodes_unlikely=trial.suggest_categorical('dead_nodes_unlikely', [False, True]),
                         step=trial.suggest_float('step', 2/8, 2*2, log=True),
                         eps=trial.suggest_float('eps', 0.0581/4, 0.0581*4, log=True),
                         nu=trial.suggest_float('nu', 0.0031/4, 0.0031*4, log=True),
                         M=trial.suggest_int('M', 745/2, 745),
                         num_grad_q=trial.suggest_int('num_grad_Q', 1, 2),
                         log_level=2,
                         check_dt=True
                     ),
            stim_delay=offset*d.neural_data.dt, attempt_correction=True)
        pred_error = ArrayWithTime.from_list(sr.log['pred_error'], drop_early_nans=True, squeeze_type='to_2d')
    except ValueError:
        return np.nan, np.nan, np.nan

    s = slice(d.end_of_visual_period_time, None)

    entropy = float(np.nanmean(ArrayWithTime.from_list(sr.autoreg.log['entropy']).slice_by_time(s)))
    lpp = float(np.nanmean(ArrayWithTime.from_list(sr.autoreg.log['log_pred_p']).slice_by_time(s)))
    one_step_error = float(np.nanmean(pred_error ** 2))

    return one_step_error, lpp, entropy




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( "--reset-db", required=False, default=False, action="store_true")
    args = parser.parse_args()

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    if args.reset_db:
        import os
        os.system('mysql -u jgould -ppassword -e "DROP DATABASE example"')
        os.system('mysql -u jgould -ppassword -e "CREATE DATABASE IF NOT EXISTS example"')

    study_name = "example"  # Unique identifier of the study.
    storage_name = f"mysql://jgould:password@127.0.0.1/{study_name}" # DROP DATABASE example
    study = optuna.create_study(study_name=study_name, storage=storage_name, directions=['minimize', 'maximize', 'minimize'], load_if_exists=True)

    if not args.reset_db:
        study.optimize(objective, n_trials=2_000)
