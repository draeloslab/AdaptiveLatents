import numpy as np
import pytest
import adaptive_latents
from adaptive_latents.profiling_functions import get_speed_by_time, get_speed_per_step

longrun = pytest.mark.skipif("not config.getoption('longrun')")

def test_fast_enough_for_resampled_buzaki_data():
    obs, beh = adaptive_latents.input_sources.hmm_simulation.simulate_example_data(n=200)
    bin_width = 0.03

    times = get_speed_per_step(psvd_input=obs, regression_output=beh)
    step_times = np.sum(list(times.values()), axis=0)
    assert np.quantile(step_times, .99) < bin_width  # 30 ms is a pretty standard bin width


def test_get_speed_by_time_works():
    obs, beh = adaptive_latents.input_sources.hmm_simulation.simulate_example_data(n=200)

    get_speed_by_time(psvd_input=obs, regression_output=beh)