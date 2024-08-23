import timeit
import numpy as np
from adaptive_latents import Bubblewrap, VanillaOnlineRegressor
from adaptive_latents.prosvd import BaseProSVD
from .jpca import BaseSJPCA
from adaptive_latents.regressions import SemiRegularizedRegressor


def get_speed_per_step(psvd_input, regression_output, prosvd_k=6, bw_params=None, max_steps=10_000):
    # todo: try transposing `obs`
    psvd = BaseProSVD(prosvd_k)
    jpca = BaseSJPCA()

    bw_params = bw_params if bw_params is not None else {}
    bw_params['go_fast'] = True
    bw = Bubblewrap(prosvd_k, **bw_params)

    reg = SemiRegularizedRegressor(input_d=bw.N, output_d=regression_output.shape[1])

    prosvd_init = 20
    psvd.initialize(psvd_input[:prosvd_init].T)

    jpca_init = 1
    o = psvd_input[prosvd_init+1]
    psvd.updateSVD(o[:, None])
    o = psvd.project_down(o[:, None]).T
    jpca.initialize(o)

    # initial observations
    start_index = prosvd_init + jpca_init
    end_index = start_index + bw.M
    for i in range(start_index, end_index):
        o = psvd_input[i]
        if np.any(np.isnan(o)):
            continue
        psvd.updateSVD(o[:, None])
        o = psvd.project_down(o[:, None]).T

        jpca.observe(o)
        o = jpca.project(o)
        bw.observe(o)

    # initialize the model
    bw.init_nodes()
    bw.e_step()
    bw.grad_Q()

    # run online
    start_index = prosvd_init + bw.M
    end_index = min(start_index + max_steps, psvd_input.shape[0])

    times = {"prosvd": [], "jpca": [], "bubblewrap": [], "regression": [], "regression prediction": []}
    for i in range(start_index, end_index):
        o = psvd_input[i]
        if np.any(np.isnan(o)):
            continue

        start_time = timeit.default_timer()
        # prosvd update
        psvd.updateSVD(o[:, None])
        o = psvd.project_down(o[:, None]).T
        end_time = timeit.default_timer()
        times["prosvd"].append(end_time - start_time)

        start_time = timeit.default_timer()
        jpca.observe(o)
        o = jpca.project(o)
        end_time = timeit.default_timer()
        times["jpca"].append(end_time - start_time)

        # bubblewrap update
        start_time = timeit.default_timer()
        bw.observe(o)
        bw.e_step()
        bw.grad_Q()
        end_time = timeit.default_timer()
        times["bubblewrap"].append(end_time - start_time)

        # regression update
        start_time = timeit.default_timer()
        reg.observe(np.array(bw.alpha), regression_output[i])
        end_time = timeit.default_timer()
        times["regression"].append(end_time - start_time)

        start_time = timeit.default_timer()
        reg.predict(np.array(bw.alpha @ bw.A))
        end_time = timeit.default_timer()
        times["regression prediction"].append(end_time - start_time)

    times = {k: np.array(ts) for k, ts in times.items()}

    return times


def get_speed_by_time(psvd_input, regression_output, prosvd_k=6, bw_params=None, max_steps=10_000):
    psvd = BaseProSVD(prosvd_k)
    jpca = BaseSJPCA()

    bw_params = bw_params if bw_params is not None else {}
    bw_params['go_fast'] = True
    bw = Bubblewrap(prosvd_k, **bw_params)

    reg = SemiRegularizedRegressor(input_d=bw.N, output_d=regression_output.shape[1])

    prosvd_init = 20
    psvd.initialize(psvd_input[:prosvd_init].T)


    jpca_init = 1
    o = psvd_input[prosvd_init+1]
    psvd.updateSVD(o[:, None])
    o = psvd.project_down(o[:, None]).T
    jpca.initialize(o)

    # initial observations
    start_index = prosvd_init + jpca_init
    end_index = start_index + bw.M
    for i in range(start_index, end_index):
        o = psvd_input[i]
        if np.any(np.isnan(o)):
            continue
        psvd.updateSVD(o[:, None])
        o = psvd.project_down(o[:, None]).T

        jpca.observe(o)
        o = jpca.project(o)
        if np.any(np.isnan(o)):
            continue
        bw.observe(o)

    # initialize the model
    bw.init_nodes()
    bw.e_step()
    bw.grad_Q()

    # run online
    start_index = prosvd_init + bw.M
    end_index = min(start_index + max_steps, psvd_input.shape[0])

    times = []
    for i in np.arange(start_index, end_index):
        times.append(timeit.default_timer())
        o = psvd_input[i]
        if np.any(np.isnan(o)):
            continue

        # prosvd update
        psvd.updateSVD(o[:, None])
        o = psvd.project_down(o[:, None]).T

        jpca.observe(o)
        o = jpca.project(o)

        if np.any(np.isnan(o)):
            continue

        # bubblewrap update
        bw.observe(o)
        bw.e_step()
        bw.grad_Q()

        # regression update
        if not np.any(np.isnan(regression_output[i])):
            reg.observe(np.array(bw.alpha), regression_output[i])

        reg.predict(np.array(bw.alpha @ bw.A))
    times.append(timeit.default_timer())

    print(psvd.Q)
    print(jpca.last_U)
    print(bw.alpha)
    print(reg.get_beta())

    times = np.array(times)

    return times
