import pytest
import numpy as np
from adaptive_latents import Bubblewrap, BWRun, SymmetricNoisyRegressor, default_clock_parameters
from adaptive_latents.input_sources.data_sources import NumpyTimedDataSource

@pytest.fixture
def outdir(tmpdir):
    tmpdir.mkdir("generated")
    outdir = tmpdir.mkdir("generated/bubblewrap_runs")
    return outdir


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def make_br(rng, outdir):
    def br_f():
        m, n_obs, n_beh = 200, 2, 3
        obs_ds = NumpyTimedDataSource(rng.normal(size=(m, n_obs)), timepoints=None, time_offsets=(-3,0,3))
        beh_ds = NumpyTimedDataSource(rng.normal(size=(m, n_beh)), timepoints=None, time_offsets=(-3,0,3))

        bw = Bubblewrap(n_obs, **default_clock_parameters)
        reg = SymmetricNoisyRegressor(bw.N, n_beh)
        br = BWRun(bw, obs_ds, beh_ds, behavior_regressor=reg, show_tqdm=True, output_directory=outdir)
        br.run(save=True)
        return br
    return br_f
