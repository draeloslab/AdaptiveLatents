import numpy as np
import matplotlib.pyplot as plt

import adaptive_latents
from adaptive_latents import Bubblewrap, ArrayWithTime, StreamingKalmanFilter
from adaptive_latents.plotting_functions import MultiRowRunComparison

def test_compare_bw_runs(rng, show_plots):
    bws = []
    for _ in range(2):
        bw = Bubblewrap(num=10, M=10, log_level=2, check_dt=True)
        hmm = adaptive_latents.input_sources.hmm_simulation.HMM.gaussian_clock_hmm()
        states, observations = hmm.simulate_with_states(n_steps=50, rng=rng)
        bw.offline_run_on(observations)
        bws.append(bw)

    MultiRowRunComparison.compare_bw_runs(bws)
    if show_plots:
        plt.show(block=True)

    behavior_dicts = [{'predicted_behavior': ArrayWithTime.from_notime(rng.normal(size=(20,3))), 'true_behavior':ArrayWithTime.from_notime(rng.normal(size=(10,3)))} for _ in range(len(bws))]
    MultiRowRunComparison.compare_bw_runs(bws, behavior_dicts)
    if show_plots:
        plt.show(block=True)

def test_compare_predictor_runs(rng, show_plots):
    predictors = []

    hmm = adaptive_latents.input_sources.hmm_simulation.HMM.gaussian_clock_hmm()
    states, observations = hmm.simulate_with_states(n_steps=50, rng=rng)

    bw = Bubblewrap(num=10, M=10, log_level=2, check_dt=True)
    bw.offline_run_on(observations)
    predictors.append(bw)

    kf = StreamingKalmanFilter(log_level=2, check_dt=True)
    kf.offline_run_on(observations)
    predictors.append(kf)


    MultiRowRunComparison.compare_predictor_runs(predictors)
    if show_plots:
        plt.show(block=True)
