import timeit
import numpy as np
from adaptive_latents import default_rwd_parameters, Bubblewrap, VanillaOnlineRegressor, proSVD
from adaptive_latents.regressions import SemiRegularizedRegressor


def get_speed_per_step(psvd_input, regression_output, prosvd_k=6, bw_params=None, max_steps=10_000):
    # todo: try transposing `obs`
    psvd = proSVD(prosvd_k)
    bw = Bubblewrap(prosvd_k, **dict(default_rwd_parameters, go_fast=True, **(bw_params if bw_params is not None else {})))
    reg = SemiRegularizedRegressor(input_d=bw.N, output_d=regression_output.shape[1])

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

    times = {"prosvd":[], "bubblewrap":[], "regression":[], "regression prediction":[]}
    for i in range(start_index, end_index):
        o = psvd_input[i]

        start_time = timeit.default_timer()
        # prosvd update
        psvd.updateSVD(o[:, None])
        o = o @ psvd.Q
        end_time = timeit.default_timer()
        times["prosvd"].append(end_time-start_time)

        # bubblewrap update
        start_time = timeit.default_timer()
        bw.observe(o)
        bw.e_step()
        bw.grad_Q()
        end_time = timeit.default_timer()
        times["bubblewrap"].append(end_time-start_time)

        # regression update
        start_time = timeit.default_timer()
        reg.observe(np.array(bw.alpha), regression_output[i])
        end_time = timeit.default_timer()
        times["regression"].append(end_time-start_time)

        start_time = timeit.default_timer()
        reg.predict(np.array(bw.alpha @ bw.A))
        end_time = timeit.default_timer()
        times["regression prediction"].append(end_time-start_time)

    times = {k:np.array(ts) for k, ts in times.items()}

    return times


def get_speed_by_time(psvd_input, regression_output, prosvd_k=6, bw_params=None, max_steps=10_000):
    psvd = proSVD(prosvd_k)
    bw = Bubblewrap(prosvd_k, **dict(default_rwd_parameters, go_fast=True, **(bw_params if bw_params is not None else {})))
    reg = SemiRegularizedRegressor(input_d=bw.N, output_d=regression_output.shape[1])

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
    times.append(timeit.default_timer())
    for i in np.arange(start_index, end_index):
        o = psvd_input[i]

        # prosvd update
        psvd.updateSVD(o[:, None])
        o = o @ psvd.Q

        # bubblewrap update
        bw.observe(o)
        bw.e_step()
        bw.grad_Q()

        # regression update
        reg.observe(np.array(bw.alpha), regression_output[i])
        reg.predict(np.array(bw.alpha @ bw.A))

        times.append(timeit.default_timer())

    times = np.array(times)

    return times
