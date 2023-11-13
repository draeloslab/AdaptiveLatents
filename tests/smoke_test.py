from bubblewrap import Bubblewrap
from bubblewrap.default_parameters import default_clock_parameters
from bubblewrap.input_sources.data_sources import NumpyPairedDataSource, NumpyTimedDataSource
from bubblewrap.bw_run import BWRun, AnimationManager
import bubblewrap.plotting_functions as bpf
from bubblewrap.regressions import SymmetricNoisyRegressor


def test_can_run_with_beh(rng, outdir):
    m, n_obs, n_beh = 150, 3, 4
    obs = rng.normal(size=(m, n_obs))
    beh = rng.normal(size=(m, n_beh))
    obs_ds = NumpyTimedDataSource(obs, None, (0,1))
    beh_ds = NumpyTimedDataSource(beh, None, (0,1))

    bw = Bubblewrap(n_obs, **default_clock_parameters)
    reg = SymmetricNoisyRegressor(bw.N, n_beh)
    br = BWRun(bw, obs_ds=obs_ds, beh_ds=beh_ds, behavior_regressor=reg, show_tqdm=False, output_directory=outdir)
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
            bpf.show_A(self.ax, bw) # note: you would usually index into ax, but this call uses 1 row and 1 column

    ca = CustomAnimation()

    bw = Bubblewrap(3, **default_clock_parameters)
    reg = SymmetricNoisyRegressor(bw.N, n_beh)
    br = BWRun(bw, obs_ds, beh_ds, behavior_regressor=reg, animation_manager=ca, show_tqdm=False, output_directory=outdir)
    br.run()

def test_can_use_cuda():
    from jax.lib import xla_bridge
    assert xla_bridge.get_backend().platform == 'gpu'

# TODO: test different regressors work together