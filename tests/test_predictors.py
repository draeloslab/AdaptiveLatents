import copy
import functools

import numpy as np
import pytest

import adaptive_latents
from adaptive_latents import VJF, ArrayWithTime, Bubblewrap
from adaptive_latents.input_sources import AR_K, LDS, KalmanFilter
from adaptive_latents.input_sources.kalman_filter import StreamingKalmanFilter

longrun = pytest.mark.skipif("not config.getoption('longrun')")


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
    lds = LDS.circular_lds(transitions_per_rotation=trasitions_per_rotation, obs_center=10, obs_noise=0, obs_d=2)
    Y, X, _ = lds.simulate(12*60, initial_state=[0, 5], rng=rng)

    kf = KalmanFilter(use_steady_state_k=use_steady_state_k)
    kf.fit(X, Y)

    check_lds_predicts_circle(kf, X, trasitions_per_rotation, show_plots)


@pytest.mark.parametrize('rank_limit', [2, None])
def test_ar_k(rng, rank_limit, show_plots):
    trasitions_per_rotation = 60
    lds = LDS.circular_lds(transitions_per_rotation=trasitions_per_rotation, obs_center=10, obs_noise=0, obs_d=2)
    _, X, stim = lds.simulate(5*60, initial_state=[0, 5],  rng=rng)

    ar = AR_K(k=1, rank_limit=rank_limit, init_method='full_rank', iter_limit=500, rng=rng)
    ar.fit(X, stim)

    check_lds_predicts_circle(ar, X, trasitions_per_rotation, show_plots)


@longrun
@pytest.mark.parametrize('predictor_maker,n_rotations', [
    (StreamingKalmanFilter, 10),
    (functools.partial(Bubblewrap), 250),
    (functools.partial(VJF, latent_d=2, rng=np.random.default_rng(4)), 1000),
])
def test_predictor_accuracy(predictor_maker, n_rotations, rng, show_plots):
    transitions_per_rotation = 30
    radius = 10
    _, Y, _ = LDS.run_nest_dynamical_system(rotations=n_rotations, transitions_per_rotation=transitions_per_rotation, radius=radius, u_function=lambda **_: np.zeros(3), rng=rng)

    predictor: adaptive_latents.transformer.StreamingTransformer = predictor_maker()

    predictor.offline_run_on([(Y, 'X')], convinient_return=False)

    trajectory = []
    for i in range(0, transitions_per_rotation+2):  # TODO: what's the correct number of transitions here? +1 or +2?
        stream = 'dt_X'
        prediction = predictor.partial_fit_transform(ArrayWithTime([[i]], Y.t[-1]), stream=stream)
        trajectory.append(prediction)

    assert not np.isclose(trajectory[1].t, Y.t[-1] + Y.dt)
    assert np.isclose(trajectory[1].t, Y.t[-1])

    trajectory = np.squeeze(trajectory)

    if show_plots:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(Y[:, 0], Y[:, 1])
        ax.plot([Y[-1, 0], trajectory[0, 0]], [Y[-1, 1], trajectory[0, 1]], '--.', color='C2')
        ax.plot(trajectory[:, 0], trajectory[:, 1], '.-', color='C1')
        ax.axis('equal')
        plt.show(block=True)

        if isinstance(predictor, Bubblewrap):
            fig, ax = plt.subplots()
            ax.plot(Y[:, 0], Y[:, 1])
            ax.plot([Y[-1, 0], trajectory[0, 0]], [Y[-1, 1], trajectory[0, 1]], '--.', color='C2')
            ax.plot(trajectory[:, 0], trajectory[:, 1], '.-')
            ax.axis('equal')
            predictor.show_bubbles_2d(ax)
            plt.show(block=True)

    half_idx = len(trajectory) // 2
    assert np.abs((np.atan2(trajectory[-1, 1], trajectory[-1, 0]) - np.atan2(Y[-1, 1], Y[-1, 0])) * 180 / np.pi) < 90  # TODO: make this tighter than 90 degrees
    assert np.abs((np.atan2(trajectory[half_idx, 1], trajectory[half_idx, 0]) - np.atan2(Y[-1, 1], Y[-1, 0])) * 180 / np.pi) > 110



@pytest.mark.parametrize('predictor_maker', [
    StreamingKalmanFilter,
    functools.partial(Bubblewrap, M=60),
    functools.partial(VJF, latent_d=2, rng=np.random.default_rng(4)),
])
def test_can_turn_off_parameter_learning(predictor_maker, rng):
    transitions_per_rotation = 30
    radius = 10
    _, Y, _ = LDS.run_nest_dynamical_system(rotations=10, transitions_per_rotation=transitions_per_rotation, radius=radius,
                                            u_function=lambda **_: np.zeros(3), rng=rng)

    Y1, Y2, Y3 = (
        Y.slice(slice(None, -2*transitions_per_rotation)),
        Y.slice(slice(-2*transitions_per_rotation, -1*transitions_per_rotation)),
        Y.slice(slice(-1*transitions_per_rotation, None)),
    )

    predictor: adaptive_latents.predictor.Predictor = predictor_maker()
    predictor.offline_run_on([(Y1, 'X')], convinient_return=False)

    dynamics_param = copy.deepcopy(predictor.get_arbitrary_dynamics_parameter())

    predictor.toggle_parameter_fitting(False)
    predictor.offline_run_on([(Y2, 'X')], convinient_return=False)
    assert (dynamics_param == predictor.get_arbitrary_dynamics_parameter()).all()

    predictor.toggle_parameter_fitting(True)
    predictor.offline_run_on([(Y3, 'X')], convinient_return=False)
    assert np.isclose(dynamics_param, predictor.get_arbitrary_dynamics_parameter()).mean() < .25