from adaptive_latents import datasets

import prosvd_code
import sjpca_code
import mmica_code

from make_step_time_figure import make_step_time_figure
from make_stability_figure import make_stability_figure
from make_nearness_to_offline_figure import make_nearness_to_offline_figure

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
    example_data = datasets.Odoherty21Dataset().neural_data

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
        case _:
            raise ValueError()

    fig.savefig(args.output, bbox_inches="tight")
