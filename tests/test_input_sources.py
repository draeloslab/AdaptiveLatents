import numpy as np
import adaptive_latents.input_sources as ins
import pytest

longrun = pytest.mark.skipif("not config.getoption('longrun')")

# @pytest.fixture(params=["numpy"])
# def ds(rng, request):
#     m = 500
#     if request.param == "numpy":
#         t = np.linspace(0,30*np.pi, m)
#         obs = np.vstack([np.sin(t), np.cos(t)]).T
#         return NumpyPairedDataSource(obs, np.mod(t, np.pi), time_offsets=(-10, -1, 0, 10))
#
# # todo: test with no time_offset
#
# def test_can_run(ds):
#     ds: ConsumableDataSource
#     assert hasattr(ds, "output_shape")
#     assert type(ds.output_shape) == tuple
#     for idx, _ in enumerate(ds.triples(1e4)):
#         assert idx < 600
#         # assert idx < 500
#         for t in ds.time_offsets:
#             ds.get_atemporal_data_point(offset=t)
#         ds.get_history()
#
#     # assert idx == 499
#
# def test_history_is_correct(ds):
#     ds: ConsumableDataSource
#     last_obs, last_beh = None, None
#     for idx, (obs, beh, offset_pairs) in enumerate(ds.triples()):
#         if idx > 0:
#             historical_obs, historical_beh = ds.get_atemporal_data_point(offset = -1)
#             assert np.all(last_obs == historical_obs)
#             assert np.all(last_beh == historical_beh)
#
#             historical_obs, historical_beh = offset_pairs[-1]
#             assert np.all(last_obs == historical_obs)
#             assert np.all(last_beh == historical_beh)
#
#             curr_obs, curr_beh = ds.get_atemporal_data_point(offset = 0)
#             assert np.all(obs == curr_obs)
#             assert np.all(beh == curr_beh)
#         last_obs = obs
#         last_beh = beh

def test_can_load_file():
    obs, beh = ins.functional.get_from_saved_npz("jpca_reduced_sc.npz")

@longrun
def test_can_load_fly():
    for identifier in ins.datasets.individual_identifiers["fly"]:
        obs, raw_behavior, obs_t, beh_t = ins.datasets.construct_fly_data(individual_identifier=identifier)


@longrun
def test_can_load_buzaki():
    for identifier in ins.datasets.individual_identifiers["buzaki"]:
        obs, raw_behavior, obs_t, beh_t = ins.datasets.construct_buzaki_data(individual_identifier=identifier, bin_width=.03)


@longrun
def test_can_load_indy():
    for identifier in ins.datasets.individual_identifiers["indy"]:
        obs, raw_behavior, obs_t, beh_t = ins.datasets.construct_indy_data(individual_identifier=identifier, bin_width=.03)

@longrun
def test_can_load_musal():
    obs, raw_behavior, obs_t, beh_t = ins.datasets.generate_musal_dataset()
