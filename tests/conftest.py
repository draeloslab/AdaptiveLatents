import pytest
import numpy as np
import adaptive_latents
from adaptive_latents import Bubblewrap, BWRun, SymmetricNoisyRegressor, NumpyTimedDataSource

# https://stackoverflow.com/a/43938191
def pytest_addoption(parser):
    parser.addoption('--longrun', action='store_true', dest="longrun",
                 default=False, help="enable 'longrun' decorated tests")

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
        m, n_obs, n_beh = 200, 2, 1
        hmm = adaptive_latents.input_sources.hmm_simulation.HMM.gaussian_clock_hmm(n_states=12, p1=.5, variance_scale=1, radius=10)
        states, observations = hmm.simulate_with_states(m, rng)
        obs_ds = NumpyTimedDataSource(observations, timepoints=np.arange(m), time_offsets=(-3,0,3))
        beh_ds = NumpyTimedDataSource(states, timepoints=np.arange(m), time_offsets=(-3,0,3))

        bw = Bubblewrap(n_obs, **adaptive_latents.default_parameters.default_clock_parameters)
        reg = SymmetricNoisyRegressor(bw.N, n_beh)
        br = BWRun(bw, obs_ds, beh_ds, behavior_regressor=reg, show_tqdm=True, output_directory=outdir)
        br.run(save=True)
        return br
    return br_f
