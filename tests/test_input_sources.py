import adaptive_latents.input_sources as ins

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
