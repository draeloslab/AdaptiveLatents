import timeit
import numpy as np
import adaptive_latents.input_sources as ins
from adaptive_latents import default_rwd_parameters, Bubblewrap, SymmetricNoisyRegressor
from proSVD import proSVD

def get_speed_over_time(psvd_input, regression_output, prosvd_k=6, bw_params=None, max_steps=10_000):
    # todo: try transposing `obs`
    psvd = proSVD(prosvd_k)
    bw = Bubblewrap(prosvd_k, **dict(default_rwd_parameters, **(bw_params if bw_params is not None else {})))
    reg = SymmetricNoisyRegressor(input_d=bw.N, output_d=regression_output.shape[1])

    prosvd_init = 20
    psvd.initialize(psvd_input[:prosvd_init].T)

    # initial observations
    start_index = prosvd_init
    end_index = start_index + bw.M
    for i in range(start_index, end_index):
        o = psvd_input[i]
        psvd.updateSVD(o[:, None])
        o = np.squeeze(o @ psvd.Q)
        bw.observe(o)

    # initialize the model
    bw.init_nodes()
    bw.e_step()
    bw.grad_Q()

    # run online
    start_index = prosvd_init + bw.M
    end_index = min(start_index + max_steps, psvd_input.shape[0])

    times = []
    for i in range(start_index, end_index):
        o = psvd_input[i]

        start_time = timeit.default_timer()
        # prosvd update
        psvd.updateSVD(o[:, None])
        o = o @ psvd.Q

        # bubblewrap update
        bw.observe(o)
        bw.e_step()
        bw.grad_Q()

        # regression update
        reg.safe_observe(np.array(bw.alpha), regression_output[i])
        reg.predict(np.array(bw.alpha @ bw.A))

        end_time = timeit.default_timer()
        times.append(end_time - start_time)


    return times


def test_fast_enough_for_resampled_buzaki_data():
    identifier = ins.datasets.individual_identifiers["buzaki"][0]
    bin_width = 0.03
    obs, position_data, obs_t, position_data_t = ins.datasets.construct_buzaki_data(individual_identifier=identifier, bin_width=bin_width)

    position_data = ins.functional.resample_behavior(raw_behavior=position_data, bin_centers=obs_t, t=position_data_t)
    position_data = position_data[:,:2]

    step_times = get_speed_over_time(psvd_input=obs, regression_output=position_data)

    assert np.quantile(step_times, .99) < bin_width * .5
