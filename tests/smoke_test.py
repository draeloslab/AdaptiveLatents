import pytest

from adaptive_latents import Bubblewrap
from adaptive_latents.default_parameters import default_clock_parameters
from adaptive_latents.input_sources.timed_data_source import NumpyTimedDataSource
from adaptive_latents.bw_run import BWRun, AnimationManager
import adaptive_latents.plotting_functions as bpf
from adaptive_latents.regressions import SymmetricNoisyRegressor
from scripts.main import main # todo: remove this?
import pickle
import numpy as np

def test_can_use_cuda():
    from jax.lib import xla_bridge
    assert xla_bridge.get_backend().platform == 'gpu'

def test_can_run_with_beh(rng, outdir):
    m, n_obs, n_beh = 150, 3, 4
    obs = rng.normal(size=(m, n_obs))
    beh = rng.normal(size=(m, n_beh))
    obs_ds = NumpyTimedDataSource(obs, None, (0,1))
    beh_ds = NumpyTimedDataSource(beh, None, (0,1))

    bw = Bubblewrap(n_obs, **default_clock_parameters)
    reg = SymmetricNoisyRegressor(bw.N, n_beh)
    br = BWRun(bw, in_ds=obs_ds, out_ds=beh_ds, behavior_regressor=reg, show_tqdm=False, output_directory=outdir)
    br.run()

def test_can_run_without_beh(rng, outdir):
    m, n_obs, n_beh = 150, 3, 4
    obs = rng.normal(size=(m, n_obs))
    obs_ds = NumpyTimedDataSource(obs, None, (0,1))

    bw = Bubblewrap(3, **default_clock_parameters)
    br = BWRun(bw, obs_ds, show_tqdm=False, output_directory=outdir)
    br.run()

def test_can_make_video(rng, outdir):
    m, n_obs, n_beh = 150, 3, 4
    obs = rng.normal(size=(m, n_obs))
    beh = rng.normal(size=(m, n_beh))
    obs_ds = NumpyTimedDataSource(obs, None, (0,1))
    beh_ds = NumpyTimedDataSource(beh, None, (0,1))

    class CustomAnimation(AnimationManager):
        n_rows = 1
        n_cols = 1
        outfile = outdir / "movie.mp4"
        def custom_draw_frame(self, step, bw, br):
            bpf.show_A(self.ax[0,0], self.fig, bw)

    ca = CustomAnimation()

    bw = Bubblewrap(3, **default_clock_parameters)
    reg = SymmetricNoisyRegressor(bw.N, n_beh)
    br = BWRun(bw, obs_ds, beh_ds, behavior_regressor=reg, animation_manager=ca, show_tqdm=False, output_directory=outdir)
    br.run()

def test_run_main(outdir):
    main(output_directory=outdir, steps_to_run=100)


def test_can_save_and_freeze(rng, outdir):
    # note I'm also passing save_A
    m, n_obs, n_beh = 300, 3, 4
    obs = rng.normal(size=(m, n_obs))
    obs_ds = NumpyTimedDataSource(obs, None, (0,1))
    bw = Bubblewrap(3, **default_clock_parameters)
    br = BWRun(bw, obs_ds, show_tqdm=False, output_directory=outdir, save_A=True)
    br.run(limit=100, save=True, freeze=True)

    pickle_file = br.pickle_file
    del br

    with open(pickle_file, 'br') as fhan:
        br = pickle.load(fhan)

    assert type(br.bw.A) == np.ndarray

    with pytest.raises(Exception):
        br.run()

def test_can_save_and_rerun(rng, outdir):
    m, n_obs, n_beh = 300, 3, 4
    obs = rng.normal(size=(m, n_obs))
    obs_ds = NumpyTimedDataSource(obs, None, (0, 1))

    bw = Bubblewrap(3, **default_clock_parameters)
    br = BWRun(bw, obs_ds, show_tqdm=False, output_directory=outdir)
    br.run(limit=100,save=True, freeze=False)

    pickle_file = br.pickle_file
    del br

    with open(pickle_file, 'br') as fhan:
        br = pickle.load(fhan)

    assert not type(br.bw.A) == np.ndarray

    br.run()

def test_if_new_method_equals_old(premade_br):
    br = premade_br
    for variable in ['alpha', 'A', 'mu', 'L', 'B', 'L_lower', 'L_diag']:
        assert np.all(br.model_step_variable_history[variable][-1] == br.bw.__dict__[variable])

    for variable in ['A', 'mu', 'L', 'B', 'L_lower', 'L_diag']:
        assert np.array_equal(br.__dict__[f"{variable}_history"], br.model_step_variable_history[variable], equal_nan=True)

    for offset in br.input_ds.time_offsets:
        assert np.array_equal(br.alpha_history[offset], br.model_offset_variable_history["alpha_prediction"][offset], equal_nan=True)
        assert np.array_equal(br.prediction_history[offset], br.model_offset_variable_history["log_pred_p"][offset], equal_nan=True)
        assert np.array_equal(br.entropy_history[offset], br.model_offset_variable_history["entropy"][offset], equal_nan=True)

    # test the h variable
    for variable in ['A', 'mu', 'L', 'B', 'L_lower', 'L_diag']:
        assert np.array_equal(br.h.__dict__[variable], br.model_step_variable_history[variable], equal_nan=True)

    for offset in br.input_ds.time_offsets:
        assert np.array_equal(br.h.alpha_prediction[offset], br.model_offset_variable_history["alpha_prediction"][offset], equal_nan=True)
        assert np.array_equal(br.h.log_pred_p[offset], br.model_offset_variable_history["log_pred_p"][offset], equal_nan=True)
        assert np.array_equal(br.h.entropy[offset], br.model_offset_variable_history["entropy"][offset], equal_nan=True)

def test_some_br_methods_run(premade_br):
    # todo: these might be deleted?
    premade_br.entropy_summary(offset=1)
    premade_br.log_pred_p_summary(offset=1)

    reg = SymmetricNoisyRegressor(input_d=premade_br.bw.A.shape[0], output_d=1)
    premade_br.evaluate_regressor(reg)


# TODO:
#  test different regressors work together
#  test_can_save_and_reload
#  test_nsteps_inbwrun_works_correctly
#  also tqdm flag
#  also should make the timing of logs more clear
#  test_can_save_A_and_other_logs
#  can make all this faster