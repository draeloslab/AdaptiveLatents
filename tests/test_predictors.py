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


@pytest.mark.skip
def test_kf_cov_pos_def(rng):
    """
    see the `evals = np.abs(evals)  # TODO: this should not be necessary` line in kalman_filter.py
    https://github.com/draeloslab/AdaptiveLatents/issues/30#issue-3000347528
    """
    transitions_per_rotation = 30
    radius = 10
    n_rotations = 10
    _, Y, _ = LDS.run_nest_dynamical_system(rotations=n_rotations, transitions_per_rotation=transitions_per_rotation, radius=radius, u_function=lambda **_: np.zeros(3), rng=rng, noise=0.05**2)

    kf = StreamingKalmanFilter()
    kf.offline_run_on(Y)

    evals = np.linalg.eigvals(kf.state_var)
    assert  (evals > 0).all()


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


@pytest.fixture(params=[
    pytest.param('kalman_filter', marks=()),
    pytest.param('bubblewrap', marks=longrun),
    pytest.param('VJF', marks=longrun),
])
def fitted_predictor_tuple(request, rng):
    predictor: adaptive_latents.predictor.Predictor
    match request.param:
        case 'kalman_filter':
            predictor = StreamingKalmanFilter()
            n_rotations = 10
        case 'bubblewrap':
            predictor = Bubblewrap()
            n_rotations = 250
        case 'VJF':
            predictor = VJF(latent_d=2, rng=np.random.default_rng(18))
            rng = np.random.default_rng(18)

            n_rotations = 500
        case _:
                raise ValueError()


    transitions_per_rotation = 30
    radius = 10
    n_test_rotations = 5
    _, Y, _ = LDS.run_nest_dynamical_system(rotations=n_rotations+n_test_rotations, transitions_per_rotation=transitions_per_rotation, radius=radius, u_function=lambda **_: np.zeros(3), rng=rng, noise=0.05**2)


    Y_train = Y.slice(slice(None, -n_test_rotations*transitions_per_rotation))
    Y_test = Y.slice(slice(-n_test_rotations*transitions_per_rotation,None))

    predictor.offline_run_on([(Y_train, 'X')], convinient_return=False)

    return predictor, Y_train, Y_test, transitions_per_rotation


def test_predictor_accuracy(fitted_predictor_tuple, show_plots):
    predictor, Y_train, Y_test, transitions_per_rotation = fitted_predictor_tuple

    trajectory = []
    for i in range(0, transitions_per_rotation+2):  # TODO: what's the correct number of transitions here? +1 or +2?
        stream = 'dt_X'
        prediction = predictor.partial_fit_transform(ArrayWithTime([[i]], Y_train.t[-1]), stream=stream)
        trajectory.append(prediction)

    assert not np.isclose(trajectory[1].t, Y_train.t[-1] + Y_train.dt)
    assert np.isclose(trajectory[1].t, Y_train.t[-1])

    trajectory = np.squeeze(trajectory)

    if show_plots:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(Y_train[:, 0], Y_train[:, 1])
        ax.plot([Y_train[-1, 0], trajectory[0, 0]], [Y_train[-1, 1], trajectory[0, 1]], '--.', color='C2')
        ax.plot(trajectory[:, 0], trajectory[:, 1], '.-', color='C1')
        ax.axis('equal')
        plt.show(block=True)

        if isinstance(predictor, Bubblewrap):
            fig, ax = plt.subplots()
            ax.plot(Y_train[:, 0], Y_train[:, 1])
            ax.plot([Y_train[-1, 0], trajectory[0, 0]], [Y_train[-1, 1], trajectory[0, 1]], '--.', color='C2')
            ax.plot(trajectory[:, 0], trajectory[:, 1], '.-')
            ax.axis('equal')
            predictor.show_bubbles_2d(ax)
            plt.show(block=True)

    half_idx = len(trajectory) // 2
    assert np.abs((np.atan2(trajectory[-1, 1], trajectory[-1, 0]) - np.atan2(Y_train[-1, 1], Y_train[-1, 0])) * 180 / np.pi) < 90  # TODO: make this tighter than 90 degrees
    assert np.abs((np.atan2(trajectory[half_idx, 1], trajectory[half_idx, 0]) - np.atan2(Y_train[-1, 1], Y_train[-1, 0])) * 180 / np.pi) > 110


def test_predictor_pdf(fitted_predictor_tuple, show_plots):
    predictor, Y_train, Y_test, transitions_per_rotation = fitted_predictor_tuple

    half_rotation = Y_test.slice(slice(None, transitions_per_rotation//2))

    a = Y_train[-1]
    pdf_a_to_a = predictor.unevaluated_log_pred_p(0)
    pdf_a_to_b = predictor.unevaluated_log_pred_p(transitions_per_rotation//2)
    predictor.offline_run_on([(half_rotation, 'X')], convinient_return=False)
    b = half_rotation.slice(-1)
    pdf_b_to_b = predictor.unevaluated_log_pred_p(0)
    pdf_b_to_a = predictor.unevaluated_log_pred_p(transitions_per_rotation//2)


    if show_plots:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=2, ncols=2)
        for ax, title, pdf_f in zip(axs.flatten(), ['a to a', 'a to b', 'b to a', 'b to b'], [pdf_a_to_a, pdf_a_to_b, pdf_b_to_a, pdf_b_to_b]):
            xlim = [Y_train[:, 0].min(), Y_train[:, 0].max()]
            ylim = [Y_train[:, 1].min(), Y_train[:, 1].max()]
            predictor.plot_pdf(fig, ax, pdf_f, xlim, ylim)

            ax.scatter(a[0], a[1], color='red')
            ax.scatter(b[0], b[1], color='blue')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)
        plt.show(block=True)

    assert pdf_a_to_a(a) > pdf_b_to_a(a) > pdf_a_to_b(a) >= pdf_b_to_b(a)
    assert pdf_b_to_b(b) > pdf_a_to_b(b) > pdf_b_to_a(b) >= pdf_a_to_a(b)


def test_can_turn_off_parameter_learning(fitted_predictor_tuple, rng):
    predictor, Y_train, Y_test, transitions_per_rotation = fitted_predictor_tuple

    Y2, Y3 = (
        Y_test.slice(slice(None, len(Y_test)//2)),
        Y_test.slice(slice(len(Y_test)//2, None)),
    )

    dynamics_param = copy.deepcopy(predictor.get_arbitrary_dynamics_parameter())

    predictor.toggle_parameter_fitting(False)
    predictor.offline_run_on([(Y2, 'X')], convinient_return=False)
    assert (dynamics_param == predictor.get_arbitrary_dynamics_parameter()).all()

    predictor.toggle_parameter_fitting(True)
    predictor.offline_run_on([(Y3, 'X')], convinient_return=False)
    assert not np.isclose(dynamics_param, predictor.get_arbitrary_dynamics_parameter()).all()