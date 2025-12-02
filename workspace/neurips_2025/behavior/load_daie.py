from scipy.io import loadmat
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as  pd
from copy import deepcopy

from adaptive_latents import ArrayWithTime, proSVD

def load_daie():
    f = h5py.File('/mnt/data/datasets/daie21/Daie_et_al_2020_targeted_photostim.mat')

    n_sessions = f['data']['dt_si'].shape[0]
    session_n = 0
    dt = f[f['data']['dt_si'][session_n,0]][0,0]

    rows = []

    for l_or_r in 'LR':
        n_stim_groups = f[f['data'][l_or_r][session_n,0]].shape[0]
        for stim_group_n in range(n_stim_groups):
            data = f[f[f['data'][l_or_r][session_n,0]][stim_group_n,0]][:]

            if stim_group_n == 0:
                stim_start, stim_end = np.nan, np.nan
            else:
                stim_start, stim_end = f[f[f['data']['epochs'][session_n,0]]['stim'][stim_group_n-1,0]][:,0]

            accuracies = f[f[f['data']['C'+l_or_r][session_n,0]][stim_group_n,0]][:,0]
            for trial, accuracy in zip(data,accuracies):
                rows.append(dict(l_or_r=l_or_r, stim_group_n=stim_group_n, trial=trial.T, stim_start=stim_start, stim_end=stim_end, accuracy=accuracy))
    _, n_neurons, n_timepoints = data.shape
    t = np.arange(n_timepoints) * dt

    df = pd.DataFrame(rows)
    nan_rows = df[df['stim_start'].isna()].sample(frac=1).reset_index(drop=True)
    non_nan_rows = df[df['stim_start'].notna()].sample(frac=1).reset_index(drop=True)
    df = pd.concat([nan_rows, non_nan_rows], ignore_index=True)

    def concat_rows(df):
        trials = []
        stims = []
        row_count = 0
        for idx, row in df.iterrows():
            trial = row['trial']
            if np.var(trial) < .1:
                continue
            stim_start = row['stim_start']
            if not np.isnan(stim_start):
                group_vec = np.zeros(n_stim_groups-1)
                group_vec[row['stim_group_n']-1] = 1
                stims.append(ArrayWithTime(group_vec, (stim_start//dt + 1) * dt + row_count * dt))

            trials.append(trial)
            row_count += trial.shape[0]
            trials.append(np.empty([5,n_neurons]) * np.nan)
            row_count += trials[-1].shape[0]


        A = np.vstack(trials)
        A = ArrayWithTime(A, np.arange(A.shape[0]) * dt)
        stims = ArrayWithTime.from_list(stims)
        return A, stims
    A, stims = concat_rows(df)
    stims.t = stims.t- dt/50

    return A, stims
