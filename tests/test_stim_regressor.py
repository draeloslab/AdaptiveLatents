from adaptive_latents.input_sources import LDS
from adaptive_latents import StreamingKalmanFilter, StimRegressor, ArrayWithTime, Pipeline, Bubblewrap
import numpy as np
import pytest

@pytest.fixture
def sr_s(rng):
    stim_magnitude = 20
    _, Y, stim = LDS.run_nest_dynamical_system(10, stim_magnitude=stim_magnitude, u_function='constant', rng=rng, radius=20) # early_shift

    sr1 = StimRegressor(autoreg=StreamingKalmanFilter(), attempt_correction=True, log_level=3)
    sr2 = StimRegressor(autoreg=StreamingKalmanFilter(), attempt_correction=False, log_level=3)
    sr3 = StimRegressor(autoreg=StreamingKalmanFilter(), attempt_correction=False, heed_stimuli=False, log_level=3)

    for sr in [sr1, sr2, sr3]:
        sr.offline_run_on(sources=[(stim,'stim'), (Y,'X')], convinient_return=False)
        sr.partial_fit_transform(ArrayWithTime([[1]], stim.t[-1] + stim.dt), stream='stim')

    return (sr1, sr2, sr3), stim_magnitude, stim


def test_api_compatible():
    StimRegressor.test_if_api_compatible()

def test_logs(sr_s, show_plots):
    (sr1, sr2, sr3), stim_magnitude, stim = sr_s
    real_stim_samples = stim.slice((stim > 0).any(axis=1))

    stim_utilized_error = ArrayWithTime.from_list(sr1.log['pred_error'], drop_early_nans=True, squeeze_type='to_2d')
    stim_aware_error = ArrayWithTime.from_list(sr2.log['pred_error'], drop_early_nans=True, squeeze_type='to_2d')
    stim_unaware_error = ArrayWithTime.from_list(sr3.log['pred_error'], drop_early_nans=True, squeeze_type='to_2d')

    errors = []
    mses = []
    for error in [stim_utilized_error, stim_aware_error, stim_unaware_error]:
        real_stim_samples2, stim_errors = ArrayWithTime.align_indices(real_stim_samples, error)
        # assert (real_stim_samples2 == real_stim_samples).all()
        _, dynamics_errors = ArrayWithTime.align_indices(real_stim_samples, error, complement=True)
        errors.append([stim_errors, dynamics_errors, error])

        start = 5
        mses.append([[np.mean(a[start:,2]**2), np.mean(a[start:,:2]**2)] for a in errors[-1]])
        # the `1:` is to avoid nans in one of the matrices

    # with np.printoptions(precision=3, suppress=True):
    #     print(np.array(mses))

    if show_plots:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=3, ncols=1)
        for ax, (stim_errors, dynamics_errors, full_errors) in zip(axs, errors):
            ax.plot(stim_errors.t, stim_errors, '.', ms=10)
            # ax.plot(dynamics_errors.t, dynamics_errors, '.', ms=5)
        plt.show(block=True)

    assert mses[0][0][0] < mses[2][0][0] - 10  #  stim-sample stim dimension errors
    assert mses[0][1][0] == mses[1][1][0]  #  dynamics-sample stim dimension errors
    assert mses[0][1][1] == mses[1][1][1]  #  dynamics-sample non-stim dimension errors


def test_log_pred_pdf(sr_s, show_plots):
    (sr1, sr2, sr3), stim_magnitude, stim = sr_s

    stim_utilized_pred = sr1.predict(1)
    stim_aware_pred = sr1.autoreg.predict(1)
    stim_unaware_pred = sr3.predict(1)
    assert (stim_aware_pred != stim_utilized_pred).all()
    assert (stim_aware_pred == sr2.predict(1)).all()
    assert (stim_aware_pred == sr2.autoreg.predict(1)).all()
    assert (stim_aware_pred != stim_unaware_pred).all()
    assert (stim_unaware_pred == sr3.autoreg.predict(1)).all()
    assert (stim_unaware_pred != stim_utilized_pred).all()

    stim_utilized_log_pdf = sr1.unevaluated_log_pred_p(1)
    stim_aware_log_pdf = sr2.unevaluated_log_pred_p(1)
    stim_unaware_log_pdf = sr3.unevaluated_log_pred_p(1)
    assert stim_utilized_log_pdf(stim_utilized_pred) > stim_utilized_log_pdf(stim_aware_pred)
    assert stim_aware_log_pdf(stim_utilized_pred) < stim_aware_log_pdf(stim_aware_pred)

    # this is the real (non-relative) test
    assert stim_utilized_log_pdf(stim_aware_pred + np.array([0,0,stim_magnitude])) > stim_utilized_log_pdf(stim_aware_pred)


def test_accepts_sparse_stimuli(rng):
    stim_magnitude = 20
    _, Y, stim = LDS.run_nest_dynamical_system(1, stims_per_rotation=5, stim_magnitude=stim_magnitude, u_function='constant', rng=rng, radius=20) # early_shift

    sr1 = StimRegressor(autoreg=StreamingKalmanFilter(steps_between_refits=3), attempt_correction=False, log_level=3, heed_stimuli=True)
    sr1.offline_run_on(sources=[(stim, 'stim'), (Y, 'X')])

    stim = stim.slice((stim != 0).any(axis=1))

    sr2 = StimRegressor(autoreg=StreamingKalmanFilter(steps_between_refits=3), attempt_correction=False, log_level=3, heed_stimuli=True)
    sr2.offline_run_on(sources=[(stim, 'stim'), (Y, 'X')])


    # import matplotlib.pyplot as plt
    # e1 = ArrayWithTime.from_list(sr1.log['pred_error'], drop_early_nans=False, squeeze_type='to_2d')
    # e2 = ArrayWithTime.from_list(sr2.log['pred_error'], drop_early_nans=False, squeeze_type='to_2d')
    # plt.plot(e1.t, e1-e2, '.-')
    # plt.plot(stim.t, stim.t * 0, '.')
    # plt.show(block=True)
    assert np.array_equal(np.array(sr1.log['pred_error']), np.array(sr2.log['pred_error']), equal_nan=True)



# def test_skips_steps(rng):
#     _, Y, _ = LDS.circular_lds().simulate(20)
#     Y1 = Y.slice(slice(None, 10))
#     Y2 = Y.slice(slice(10, None))
#
#
#     sr = StimRegressor(autoreg=Bubblewrap(num=10, M=5))
#     sr.offline_run_on([(Y1, 'X')])
#
#     par = sr.get_arbitrary_dynamics_parameter()
#     for i in range(10):
#         sr.partial_fit_transform(Y2.slice(slice(i,i+1)), stream='X')
#         assert not (sr.get_arbitrary_dynamics_parameter() == par).all()
#         par = sr.get_arbitrary_dynamics_parameter()


def test_not_heeding_works(rng):
    stim_magnitude = 20
    _, Y, stim = LDS.run_nest_dynamical_system(2, stim_magnitude=stim_magnitude, u_function='constant', rng=rng, radius=20) # early_shift

    sr2 = StimRegressor(autoreg=StreamingKalmanFilter(), attempt_correction=False, heed_stimuli=True, log_level=3)
    sr3 = StimRegressor(autoreg=StreamingKalmanFilter(), attempt_correction=False, heed_stimuli=False, log_level=3)
    kf = StreamingKalmanFilter(log_level=3, check_dt=True)

    for p in [sr2, sr3, kf]:
        p.offline_run_on(sources=[(stim,'stim'), (Y,'X')], convinient_return=False)
        p.partial_fit_transform(ArrayWithTime([[1]], stim.t[-1] + stim.dt), stream='stim')

    assert np.array_equal(np.array(sr3.log['pred_error']), np.array(kf.log['pred_error']), equal_nan=True)
    assert not np.array_equal(np.array(sr2.log['pred_error']), np.array(kf.log['pred_error']), equal_nan=True)
