import matplotlib.pyplot as plt
import numpy as np
from adaptive_latents import Bubblewrap, VJF, StreamingKalmanFilter
from adaptive_latents.input_sources import LDS


def make_figure(predictor, n_rotations, rng):
    """basically a clone of adaptive_latents/tests/test_predictors::test_predictor_pdf"""
    transitions_per_rotation = 30
    radius = 10
    n_test_rotations = 5
    _, Y, _ = LDS.run_nest_dynamical_system(
        rotations=n_rotations + n_test_rotations,
        transitions_per_rotation=transitions_per_rotation,
        radius=radius,
        u_function=lambda **_: np.zeros(3),
        rng=rng,
        noise=0.05**2,
    )

    Y_train = Y.slice(slice(None, -n_test_rotations * transitions_per_rotation))
    Y_test = Y.slice(slice(-n_test_rotations * transitions_per_rotation, None))

    predictor.offline_run_on([(Y_train, "X")], convinient_return=False, show_tqdm=True)

    half_rotation = Y_test.slice(slice(None, transitions_per_rotation // 2))

    a = Y_train[-1]
    pdf_a_to_a = predictor.unevaluated_log_pred_p(0)
    pdf_a_to_b = predictor.unevaluated_log_pred_p(transitions_per_rotation // 2)
    predictor.offline_run_on([(half_rotation, "X")], convinient_return=False)
    b = half_rotation.slice(-1)
    pdf_b_to_b = predictor.unevaluated_log_pred_p(0)
    pdf_b_to_a = predictor.unevaluated_log_pred_p(transitions_per_rotation // 2)

    fig, axs = plt.subplots(nrows=2, ncols=2)
    for ax, title, pdf_f in zip(axs.flatten(), ['a to a', 'a to b', 'b to a', 'b to b'], [pdf_a_to_a, pdf_a_to_b, pdf_b_to_a, pdf_b_to_b]):
        xlim = [Y_train[:, 0].min(), Y_train[:, 0].max()]
        ylim = [Y_train[:, 1].min(), Y_train[:, 1].max()]
        predictor.plot_pdf(fig, ax, pdf_f, xlim, ylim)

        ax.scatter(a[0], a[1], color='red')
        ax.scatter(b[0], b[1], color='blue')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

    return fig


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    parser.add_argument("--pred_type", type=str, required=True)

    args = parser.parse_args()

    rng = np.random.default_rng(0)
    match args.pred_type:
        case "kf":
            predictor = StreamingKalmanFilter()
            n_rotations = 10
        case "bw":
            predictor = Bubblewrap()
            n_rotations = 250
        case "vjf":
            predictor = VJF(latent_d=2, rng=np.random.default_rng(18))
            n_rotations = 500
        case _:
            raise ValueError()

    fig = make_figure(predictor, n_rotations, rng)

    fig.savefig(args.output, bbox_inches="tight")
