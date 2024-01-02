import timeit
import numpy as np
import adaptive_latents.input_sources as ins
from adaptive_latents import NumpyTimedDataSource, default_rwd_parameters, Bubblewrap, SymmetricNoisyRegressor, BWRun
from proSVD import proSVD


def test_fast_enough_for_buzaki_data():
    identifier = ins.datasets.individual_identifiers["buzaki"][0]
    # todo: try transposing `obs`
    bin_width = 0.03
    obs, _, _, _ = ins.datasets.construct_buzaki_data(individual_identifier=identifier, bin_width=bin_width)

    k = 6
    psvd = proSVD(k)
    bw = Bubblewrap(k, **default_rwd_parameters)

    prosvd_init = 20
    psvd.initialize(obs[:prosvd_init].T)

    # initial observations
    start = prosvd_init
    end = start + bw.M
    for i in range(start, end):
        o = obs[i]
        psvd.updateSVD(o[:,None])
        o = np.squeeze(o @ psvd.Q)
        bw.observe(o)

    # initialize the model
    bw.init_nodes()
    bw.e_step()
    bw.grad_Q()

    # run online
    max_steps = 10_000
    start = prosvd_init + bw.M
    end = min(start + max_steps, obs.shape[0])

    start_time = timeit.default_timer()
    for i in range(start, end):
        o = obs[i]
        psvd.updateSVD(o[:,None])
        o = o @ psvd.Q
        bw.observe(o)
        bw.e_step()
        bw.grad_Q()
    end_time = timeit.default_timer()

    elapsed_time = end_time - start_time
    print(f"{elapsed_time = } {end-start = }")
    print(f"{elapsed_time/(end-start) = } {bin_width * .5 = }")
    assert elapsed_time/(end - start) < bin_width * .5

    # reg = SymmetricNoisyRegressor(bw.N, raw_behavior.shape[1])
    # br = BWRun(bw, obs_ds=obs_ds, beh_ds=beh_ds, behavior_regressor=reg, show_tqdm=False, output_directory=outdir)
    # br.run()

if __name__ == '__main__':
    test_fast_enough_for_buzaki_data()