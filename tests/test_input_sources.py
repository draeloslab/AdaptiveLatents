import numpy as np
import pytest
from scipy.stats import special_ortho_group

import adaptive_latents.input_sources as ins


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

@pytest.mark.parametrize('rank_limit', [2, None])
def test_ar_k(rng, rank_limit, show_plots):
    if show_plots:
        import matplotlib.pyplot as plt

    samples_per_second = 20
    seconds_per_rotation = 3
    total_seconds = 5

    t = np.linspace(0, total_seconds/seconds_per_rotation * np.pi * 2, total_seconds*samples_per_second+1)
    X = np.vstack((np.cos(t), np.sin(t))).T + 10
    stim = np.zeros(shape=(X.shape[0],1))

    ar = ins.autoregressor.AR_K(k=1, rank_limit=rank_limit, init_method='full_rank', iter_limit=500, rng=rng)
    ar.fit(X, stim)

    if show_plots:
        plt.plot(X[:,0], X[:,1])

    initial_point = np.array([[10,12]])
    X_hat = ar.predict(initial_observations=initial_point, n_steps=samples_per_second*seconds_per_rotation)

    one_step_distance = np.linalg.norm(X[0]-X[1])

    if show_plots:
        plt.plot([initial_point[0,0], X_hat[0,0]], [initial_point[0,1], X_hat[0,1]], '--.', color='C1')
        plt.plot(X_hat[:,0], X_hat[:,1], '.-')

    assert close(X_hat[-1], initial_point, 2*one_step_distance)
    assert not close(X_hat[X_hat.shape[0]//2], initial_point, 2*one_step_distance)

    initial_point = np.array([[10,10]])
    X_hat = ar.predict(initial_observations=initial_point, n_steps=samples_per_second*seconds_per_rotation)
    if show_plots:
        plt.plot([initial_point[0,0], X_hat[0,0]], [initial_point[0,1], X_hat[0,1]], '--.', color='C2')
        plt.plot(X_hat[:,0], X_hat[:,1], '.-')
    assert close(X_hat[-1], initial_point, .1*one_step_distance)
    assert close(X_hat[X_hat.shape[0]//2], initial_point, .3)
    if show_plots:
        plt.axis('equal')
        plt.show()


@pytest.mark.parametrize('use_steady_state_k', [True, False])
def test_kalman_filter(rng, show_plots, use_steady_state_k):
    if show_plots:
        import matplotlib.pyplot as plt

    samples_per_second = 20
    seconds_per_rotation = 3
    total_seconds = 12

    t = np.linspace(0, total_seconds / seconds_per_rotation * np.pi * 2, total_seconds * samples_per_second + 1)
    X = np.vstack((np.cos(t), np.sin(t))).T
    X = X + 10
    Y = X @ special_ortho_group(dim=20, seed=rng).rvs()[:, :2].T

    kf = ins.kalman_filter.KalmanFilter(use_steady_state_k=use_steady_state_k)
    kf.fit(X, Y)

    if show_plots:
        plt.plot(X[:, 0], X[:, 1])

    initial_point = X[-1:]
    X_hat = kf.predict(n_steps=samples_per_second * seconds_per_rotation, initial_state=initial_point)
    one_step_distance = np.linalg.norm(X[0] - X[1])

    if show_plots:
        plt.plot([initial_point[0, 0], X_hat[0, 0]], [initial_point[0, 1], X_hat[0, 1]], '--.', color='C1')
        plt.plot(X_hat[:, 0], X_hat[:, 1], '.-')

    assert close(X_hat[-1], initial_point, 2 * one_step_distance)
    assert not close(X_hat[X_hat.shape[0] // 2], initial_point, 2 * one_step_distance)

    initial_point = np.array([[10, 10]])
    X_hat = kf.predict(initial_state=initial_point, n_steps=samples_per_second * seconds_per_rotation)

    if show_plots:
        plt.plot([initial_point[0, 0], X_hat[0, 0]], [initial_point[0, 1], X_hat[0, 1]], '--.', color='C2')
        plt.plot(X_hat[:, 0], X_hat[:, 1], '.-')
    assert close(X_hat[-1], initial_point, .1 * one_step_distance)
    assert close(X_hat[X_hat.shape[0] // 2], initial_point, .3)

    if show_plots:
        plt.axis('equal')
        plt.show()
