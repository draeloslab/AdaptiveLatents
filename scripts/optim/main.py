import numpy as np
import matplotlib.pyplot as plt
import adaptive_latents.transforms.utils as fin
import tqdm as tqdm
from adaptive_latents import NumpyTimedDataSource
import adaptive_latents.plotting_functions as bpf
import adaptive_latents.input_sources.datasets as datasets
import sklearn.decomposition
import pandas as pd
import adaptive_latents
from adaptive_latents.transforms.jpca import apply_prosvd_and_sjpca_and_cache

def smooth_columns(X, t, kernel_length=5, kernel_end=-3):
    kernel = np.exp(np.linspace(0, kernel_end, kernel_length))
    kernel /= kernel.sum()
    mode = 'valid'
    X = np.column_stack([np.convolve(kernel, column, mode) for column in X.T])
    t = np.convolve(np.hstack([[1],kernel[:-1]*0]), t, mode)
    return X, t 

obs, beh, obs_t, beh_t = adaptive_latents.input_sources.datasets.construct_jenkins_data(bin_width=0.03)
beh, beh_t = fin.resample_matched_timeseries(beh, obs_t, beh_t), obs_t
obs, obs_t = smooth_columns(obs, obs_t, kernel_length=10)
pre_datasets = {
    's(obs,4) # i o': (fin.prosvd_data(input_arr=obs, output_d=4, init_size=4, centering=True), obs_t),
    'j(s(obs,4)) # i o': (apply_prosvd_and_sjpca_and_cache(input_arr=obs, intermediate_d=4, output_d=4), obs_t),
    'beh # i o': (beh, beh_t),
}

for key, value in pre_datasets.items():
    x, x_t = fin.clip(*value)
    idx = ~np.any(np.isnan(x), axis=1)
    x, x_t = x[idx], x_t[idx]
    pre_datasets[key] = (x, x_t)


with_randoms = {}
for key, value in pre_datasets.items():
    k = key.replace("🌀", "")
    with_randoms[k] = value
    if "🌀" in key:
        k, tags = k.split("#")
        k = k.strip()
        with_randoms[f"shuf({k}) #{tags.strip()}"] = (*fin.shuffle_time(value[0]), value[1])

datasets = {}
input_keys = []
output_keys = []
for key, value in with_randoms.items():
    k, tags = key.split("#")
    k = k.strip()
    datasets[k] = value
    assert np.all(np.isfinite(value[0]))
    if "i" in tags:
        input_keys.append(k)
    if "o" in tags:
        output_keys.append(k)
        if "b" in tags:
            a, t = value
            for i in range(a.shape[1]):
                new_k = k + f"[:,{i}]"
                datasets[new_k] = (a[:,i:i+1],t)
                output_keys.append(new_k)






import optuna

def objective(trial: optuna.trial.Trial):
    bw_params = dict(    
        adaptive_latents.default_parameters.default_rwd_parameters,
        M = 100,
        num_grad_q=1,
        num=trial.suggest_int('num', 10, 1500),
        eps=trial.suggest_float('eps', 1e-10, 10, log=True),
        step=trial.suggest_float('step', 1e-10, 10, log=True),
    )


    out_ds = adaptive_latents.NumpyTimedDataSource(*datasets['j(s(obs,4))'], (1,))
    in_ds = adaptive_latents.NumpyTimedDataSource(*datasets['beh'], (0, 1))

    bw = adaptive_latents.Bubblewrap(in_ds.output_shape, **bw_params)

    reg_class = adaptive_latents.regressions.auto_regression_decorator(
        adaptive_latents.regressions.SemiRegularizedRegressor, n_steps=0, autoregress_only=False)

    reg = reg_class(input_d=bw.N, output_d=out_ds.output_shape, regularization_factor=.001)
    br = adaptive_latents.BWRun(bw, in_ds=in_ds, out_ds=out_ds, behavior_regressor=reg, show_tqdm=True)

    br.run(limit=1000, save=False, freeze=False)
    for i in range(50):
        br.run(limit=200, save=False, freeze=False)
        error = sum(br.get_last_half_metrics(offset=1)['beh_sq_error'])
        trial.report(error, step=br.bw.obs.n_obs)

        if trial.should_prune():
            raise optuna.TrialPruned()

    error = sum(br.get_last_half_metrics(offset=1)['beh_sq_error'])
    return error



study_name = "study-24-05-05-22-25"
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name, pruner=optuna.pruners.MedianPruner(), load_if_exists=True)

study.optimize(objective, n_trials=100)
