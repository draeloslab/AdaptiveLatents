import adaptive_latents.input_sources as ins
import pytest

longrun = pytest.mark.skipif("not config.getoption('longrun')")


def test_hmm_runs(rng):
    # note I do not test correctness here
    for hmm in (
        ins.hmm_simulation.HMM.gaussian_clock_hmm(n_states=10),
        ins.hmm_simulation.HMM.gaussian_clock_hmm(n_states=10, high_d_pad=10),
        ins.hmm_simulation.HMM.wandering_gaussian_clock_hmm(n_states=10),
        ins.hmm_simulation.HMM.teetering_gaussian_clock_hmm(n_states=10),
        ins.hmm_simulation.HMM.inverting_gaussian_clock_hmm(n_states=10),
        ins.hmm_simulation.HMM.discrete_clock_hmm(n_states=10),
        ins.hmm_simulation.HMM.wave_clock_hmm(n_states=11),
        ins.hmm_simulation.HMM.infinity_shape_hmm(n_states=11),
    ):
        hmm.simulate(50, rng)
        states, observations = hmm.simulate_with_states(10, rng)
        hmm.advance_one_step(rng, states[-1])

@longrun
def test_can_load_fly():
    for identifier in ins.datasets.individual_identifiers["fly"]:
        obs, raw_behavior, obs_t, beh_t = ins.datasets.construct_fly_data(individual_identifier=identifier)

@longrun
def test_can_load_indy():
    for identifier in ins.datasets.individual_identifiers["indy"]:
        obs, raw_behavior, obs_t, beh_t = ins.datasets.construct_indy_data(individual_identifier=identifier, bin_width=.03)

@longrun
def test_can_load_jenkins():
    obs, raw_behavior, obs_t, beh_t = ins.datasets.construct_jenkins_data(bin_width=.03)

@longrun
def test_can_load_nason20_dataset():
    obs, raw_behavior, obs_t, beh_t = ins.datasets.construct_nason20_data(bin_width=.03)

@longrun
def test_can_load_unpublished24():
    obs, raw_behavior, obs_t, beh_t = ins.datasets.construct_unpublished24_data()

@longrun
def test_can_load_buzaki():
    for identifier in ins.datasets.individual_identifiers["buzaki"]:
        obs, raw_behavior, obs_t, beh_t = ins.datasets.construct_buzaki_data(individual_identifier=identifier,
                                                                             bin_width=.03,
                                                                             _recalculate_cache_value=True)

# @longrun
# def test_can_load_musal():
#     obs, raw_behavior, obs_t, beh_t = ins.datasets.generate_musal_dataset(_recalculate_cache_value=True)
