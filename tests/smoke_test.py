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

def test_can_use_float64():
    import jax
    jax.config.update('jax_enable_x64', True)  # this line is mostly for documentation, the real line is in conftest
    x = jax.random.uniform(jax.random.key(0), (1,), dtype=jax.numpy.float64)
    assert x.dtype == jax.numpy.float64

# def test_can_use_float32():
#     import jax
#     # jax.config.update('jax_enable_x64', False) # this should be the default
#     x = jax.random.uniform(jax.random.key(0), (1,), dtype=jax.numpy.float64)
#     assert x.dtype != jax.numpy.float64

# def test_can_use_cpu():
#     import jax
#     jax.config.update('jax_platform_name', 'cpu')
#     from jax.lib import xla_bridge
#     assert xla_bridge.get_backend().platform == 'cpu'

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
    br = BWRun(bw, obs_ds, show_tqdm=False, output_directory=outdir, log_level=2)
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


def test_post_hoc_regression_is_correct(rng, outdir):
    m, n_obs, n_beh = 150, 3, 4
    obs = rng.normal(size=(m, n_obs))
    beh = rng.normal(size=(m, n_beh)) # TODO: refactor these names to input and output

    br1 = BWRun(
        Bubblewrap(3, **default_clock_parameters),
        NumpyTimedDataSource(obs, None, (0,1)),
        NumpyTimedDataSource(beh, None, (0,1)),
        SymmetricNoisyRegressor(default_clock_parameters['num'], n_beh),
        show_tqdm=False,
        output_directory=outdir
    )
    br1.run()

    br2 = BWRun(
        Bubblewrap(3, **default_clock_parameters),
        NumpyTimedDataSource(obs, None, (0,1)),
        show_tqdm=False,
        output_directory=outdir
    )
    br2.run()
    reg = SymmetricNoisyRegressor(br2.bw.N, n_beh)
    br2.add_regression_post_hoc(reg, NumpyTimedDataSource(beh, None, (0,1)))

    assert np.allclose(br1.h.beh_error[1], br2.h.beh_error[1], equal_nan=True)
    # TODO: the times seem to be 1 out of sync with the expected bubblewrap step times (even accounting for delay)

# TODO:
#  array shapes are correct for 1d output
#  test different regressors work together
#  test_can_save_and_reload
#  test_nsteps_inbwrun_works_correctly
#  also tqdm flag
#  also should make the timing of logs more clear
#  test_can_save_A_and_other_logs
#  can make all this faster
