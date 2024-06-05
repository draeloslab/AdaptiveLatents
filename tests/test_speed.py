import numpy as np
import adaptive_latents.input_sources as ins
import adaptive_latents
import pytest

longrun = pytest.mark.skipif("not config.getoption('longrun')")

@longrun
def test_fast_enough_for_resampled_buzaki_data():
    identifier = ins.datasets.individual_identifiers["buzaki"][0]
    bin_width = 0.03
    obs, position_data, obs_t, position_data_t = ins.datasets.construct_buzaki_data(individual_identifier=identifier, bin_width=bin_width)

    position_data = adaptive_latents.transforms.utils.resample_matched_timeseries(old_timeseries=position_data, new_sample_times=obs_t, old_sample_times=position_data_t)
    position_data = position_data[:,:2]

    times = adaptive_latents.profiling_functions.get_speed_per_step(psvd_input=obs, regression_output=position_data)
    step_times = np.sum(list(times.values()), axis=0)
    assert np.quantile(step_times, .99) < bin_width * .5


@longrun
def test_get_speed_by_time_works():
    identifier = ins.datasets.individual_identifiers["buzaki"][0]
    bin_width = 0.03
    obs, position_data, obs_t, position_data_t = ins.datasets.construct_buzaki_data(individual_identifier=identifier, bin_width=bin_width)

    position_data = adaptive_latents.transforms.utils.resample_matched_timeseries(old_timeseries=position_data, new_sample_times=obs_t, old_sample_times=position_data_t)
    position_data = position_data[:,:2]

    adaptive_latents.profiling_functions.get_speed_by_time(psvd_input=obs, regression_output=position_data)
    # TODO: add an assert