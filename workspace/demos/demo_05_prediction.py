import matplotlib.pyplot as plt

import adaptive_latents as al
from adaptive_latents import Bubblewrap, CenteringTransformer, Concatenator, KernelSmoother, Pipeline, Tee, proSVD, sjPCA

"""
Demo: Prediction
"""


def main(show_plots=True):
    d = al.datasets.Odoherty21Dataset()
    neural_data = d.neural_data
    behavioral_data = d.behavioral_data

    p = Pipeline([
        CenteringTransformer(input_streams={0: 'X'}),
        KernelSmoother(input_streams={0: 'X'}),  # smooths the neural data
        Concatenator(input_streams={0: 0, 1: 1}, output_streams={0: 0, 1: 0}),
        proSVD(k=6),
        sjPCA(),
        tea := Tee(input_streams={0: 0}),
        bw := Bubblewrap(log_level=2, input_streams={0: 'X', 2: 'dt'})
    ])

    prediction_query_times = bw.make_prediction_times(neural_data)

    p.offline_run_on([neural_data, behavioral_data, prediction_query_times], show_tqdm=True, exit_time=60)

    Bubblewrap.compare_runs([bw])
    if show_plots:
        plt.show()


if __name__ == '__main__':
    main()