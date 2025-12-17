import matplotlib.pyplot as plt

import adaptive_latents as al
from adaptive_latents import Bubblewrap, CenteringEstimator, Concatenator, KernelSmoother, Pipeline, Tee, proSVD, sjPCA
from adaptive_latents.plotting_functions import MultiRowRunComparison

"""
Demo: Prediction
"""


def main(show_plots=True):
    d = al.datasets.Odoherty21Dataset()
    neural_data = d.neural_data
    behavioral_data = d.behavioral_data

    p = Pipeline([
        CenteringEstimator(input_streams={0: 'X'}),
        KernelSmoother(input_streams={0: 'X'}),  # smooths the neural data
        Concatenator(input_streams={0: 0, 1: 1}, output_streams={0: 0, 1: 0}),
        proSVD(k=6),
        sjPCA(),
        tea := Tee(input_streams={0: 0}),
        bw := Bubblewrap(input_streams={0: 'X', 2: 'dt'}, log_level=2, check_dt=True)
    ])

    prediction_query_times = bw.make_prediction_times(neural_data)

    p.offline_run_on([neural_data, behavioral_data, prediction_query_times], show_tqdm=True, exit_time=60)

    MultiRowRunComparison.compare_bw_runs([bw])
    if show_plots:
        plt.show()


if __name__ == '__main__':
    main()