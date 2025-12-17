import numpy as np
from adaptive_latents import datasets, KernelSmoother, CenteringEstimator, Pipeline

import prosvd_code
import sjpca_code
import mmica_code
from make_step_time_figure import make_step_time_figure
from make_stability_figure import make_stability_figure
from make_nearness_to_offline_figure import make_nearness_to_offline_figure
from make_native_nearness_to_offline_figure import make_native_nearness_to_offline_figure

if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    parser.add_argument("--transformer", type=str, required=True)
    parser.add_argument("--type-of-plot", type=str, required=True)

    args = parser.parse_args()

    match args.transformer:
        case "prosvd":
            module = prosvd_code
        case "mmica":
            module = mmica_code
        case "sjpca":
            module = sjpca_code
        case _:
            raise ValueError()

    mid_d = 10
    low_d = 4

    if args.type_of_plot != "native_nearness_to_offline":
        example_data = datasets.Odoherty21Dataset().neural_data
        pipeline = Pipeline([CenteringEstimator(init_size=200, nan_when_uninitialized=True), KernelSmoother(tau=.16 / example_data.dt)])
        example_data = pipeline.offline_run_on(example_data)
    else:
        rng = np.random.default_rng(0)

    match args.type_of_plot:
        case "time":
            step_times = module.get_step_times(example_data, mid_d=mid_d, low_d=low_d)
            fig = make_step_time_figure(step_times)
        case "stability":
            Qs = module.get_projection_matrix_over_time(example_data, mid_d=mid_d, low_d=low_d)
            fig = make_stability_figure(Qs)
        case "nearness_to_offline":
            Qs = module.get_projection_matrix_over_time(example_data, mid_d=mid_d, low_d=low_d)
            offline_Q = module.get_offline_projection_matrix(example_data, mid_d=mid_d, low_d=low_d)
            fig = make_nearness_to_offline_figure(Qs, offline_Q)
        case "native_nearness_to_offline":
            outputs = module.native_nearness_to_offline(mid_d=mid_d, low_d=low_d, rng=rng, T=60, dt=0.03, iterations_to_run=10, calculate_intra_run_errors=False)
            fig = make_native_nearness_to_offline_figure(*outputs)
        case _:
            raise ValueError()

    fig.savefig(args.output, bbox_inches="tight")
