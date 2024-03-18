import pytest
import numpy as np
import adaptive_latents
from adaptive_latents import Bubblewrap, BWRun, SymmetricNoisyRegressor, NumpyTimedDataSource

# https://stackoverflow.com/a/43938191
def pytest_addoption(parser):
    parser.addoption('--longrun', action='store_true', dest="longrun",
                 default=False, help="enable 'longrun' decorated tests")

@pytest.fixture()
def outdir(tmpdir):
    tmpdir.mkdir("generated")
    outdir = tmpdir.mkdir("generated/bubblewrap_runs")
    return outdir


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _make_br(rng):
    def br_f(m=600, n_obs=2, bw_params=None, save_A=False):
        if bw_params is None:
            bw_params = dict()
        n_beh = 1

        hmm = adaptive_latents.input_sources.hmm_simulation.HMM.gaussian_clock_hmm(high_d_pad=n_obs-2)
        states, observations = hmm.simulate_with_states(m, rng)
        obs_ds = NumpyTimedDataSource(observations, timepoints=np.arange(m), time_offsets=(-3,0,1,3))
        beh_ds = NumpyTimedDataSource(states, timepoints=np.arange(m), time_offsets=(-3,0,3))

        bw = Bubblewrap(n_obs, **dict(adaptive_latents.default_parameters.default_clock_parameters, **bw_params))
        reg = SymmetricNoisyRegressor(bw.N, n_beh)
        br = BWRun(bw, obs_ds, beh_ds, behavior_regressor=reg, show_tqdm=False, save_A=save_A)
        br.run(save=False, freeze=True)
        return br
    return br_f

make_br = pytest.fixture(_make_br)

@pytest.fixture(scope="session")
def premade_br():
    return _make_br(np.random.default_rng(0))(save_A=True)

@pytest.fixture(scope="session")
def premade_big_br():
     return _make_br(np.random.default_rng(0))(n_obs=20, m=3_000)
#
