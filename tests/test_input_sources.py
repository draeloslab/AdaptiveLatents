import numpy as np
import pytest

import adaptive_latents.input_sources as ins
from adaptive_latents import VJF


def test_hmm_runs(rng):
    # note I do not test correctness here
    for hmm in (
        ins.hmm_simulation.HMM.gaussian_clock_hmm(n_states=10),
        ins.hmm_simulation.HMM.gaussian_clock_hmm(n_states=10, high_d_pad=10),
        ins.hmm_simulation.HMM.wandering_gaussian_clock_hmm(n_states=10),
        ins.hmm_simulation.HMM.teetering_gaussian_clock_hmm(n_states=10),
        ins.hmm_simulation.HMM.inverting_gaussian_clock_hmm(n_states=10),
        ins.hmm_simulation.HMM.discrete_clock_hmm(n_states=10),
        ins.hmm_simulation.HMM.wave_clock_hmm(n_states=11),
        ins.hmm_simulation.HMM.infinity_shape_hmm(n_states=11),
    ):
        hmm.simulate(50, rng)
        states, observations = hmm.simulate_with_states(10, rng)
        hmm.advance_one_step(rng, states[-1])


def close(a, b, radius):
    return np.linalg.norm(a - b) < radius


def check_lds_predicts_circle(predictor, X, trasitions_per_rotation, show_plots):
    if show_plots:
        import matplotlib.pyplot as plt

    if show_plots:
        plt.plot(X[:, 0], X[:, 1])

    initial_point = X[-1:]
    X_hat = predictor.predict(n_steps=trasitions_per_rotation, initial_state=initial_point)
    one_step_distance = np.linalg.norm(X[0] - X[1])

    if show_plots:
        plt.plot([initial_point[0, 0], X_hat[0, 0]], [initial_point[0, 1], X_hat[0, 1]], '--.', color='C1')
        plt.plot(X_hat[:, 0], X_hat[:, 1], '.-')

    assert close(X_hat[-1], initial_point, 2 * one_step_distance)
    assert not close(X_hat[X_hat.shape[0] // 2], initial_point, 2 * one_step_distance)

    initial_point = np.array([[10, 10]])
    X_hat = predictor.predict(initial_state=initial_point, n_steps=trasitions_per_rotation)

    if show_plots:
        plt.plot([initial_point[0, 0], X_hat[0, 0]], [initial_point[0, 1], X_hat[0, 1]], '--.', color='C2')
        plt.plot(X_hat[:, 0], X_hat[:, 1], '.-')
    assert close(X_hat[-1], initial_point, .1 * one_step_distance)
    assert close(X_hat[X_hat.shape[0] // 2], initial_point, .3)

    if show_plots:
        plt.axis('equal')
        plt.show()

@pytest.mark.parametrize('use_steady_state_k', [True, False])
def test_kalman_filter(rng, show_plots, use_steady_state_k):
    trasitions_per_rotation = 60
    lds = ins.LDS.circular_lds(transitions_per_rotation=trasitions_per_rotation, obs_center=10, obs_noise=0, obs_d=2)
    Y, X, _ = lds.simulate(12*60, initial_state=[0, 5], rng=rng)

    kf = ins.kalman_filter.KalmanFilter(use_steady_state_k=use_steady_state_k)
    kf.fit(X, Y)

    check_lds_predicts_circle(kf, X, trasitions_per_rotation, show_plots)


@pytest.mark.parametrize('rank_limit', [2, None])
def test_ar_k(rng, rank_limit, show_plots):
    trasitions_per_rotation = 60
    lds = ins.LDS.circular_lds(transitions_per_rotation=trasitions_per_rotation, obs_center=10, obs_noise=0, obs_d=2)
    _, X, stim = lds.simulate(5*60, initial_state=[0, 5],  rng=rng)

    ar = ins.autoregressor.AR_K(k=1, rank_limit=rank_limit, init_method='full_rank', iter_limit=500, rng=rng)
    ar.fit(X, stim)

    check_lds_predicts_circle(ar, X, trasitions_per_rotation, show_plots)


def test_VJF(rng, rank_limit, show_plots):
    trasitions_per_rotation = 30
    lds = ins.LDS.circular_lds(transitions_per_rotation=trasitions_per_rotation, obs_center=10, obs_noise=0, obs_d=2)
    X, Y, stim = lds.simulate(100*60, initial_state=[0, 5],  rng=rng)

    vjf = VJF()
    # vjf.fit(X, stim)

    check_lds_predicts_circle(vjf, X, trasitions_per_rotation, show_plots)
