import matplotlib.pyplot as plt

import adaptive_latents as al
from adaptive_latents import Pipeline, CenteringTransformer, KernelSmoother, Bubblewrap, ArrayWithTime, proSVD, sjPCA, Concatenator, VanillaOnlineRegressor

"""
Demo: Joint latent prediction and regression
"""

def main(show=True):
    d = al.datasets.Odoherty21Dataset()
    neural_data = d.neural_data
    behavioral_data = d.behavioral_data

    p = Pipeline([
        CenteringTransformer(input_streams={0: 'X'}),
        KernelSmoother(input_streams={0: 'X'}),
        Concatenator(input_streams={0: 0, 1: 1}, output_streams={0: 0, 1: 0}),
        proSVD(k=6),
        sjPCA(),
        bw := Bubblewrap(log_level=2, input_streams={0: 'X', 2: 'dt'}),
        reg := VanillaOnlineRegressor(input_streams={0: 'X', 2: 'qX', 3: 'Y'}, log_level=2),
    ])

    prediction_query_times = Bubblewrap.make_prediction_times(neural_data)

    result = p.offline_run_on(
        sources=[neural_data, behavioral_data, prediction_query_times, behavioral_data],
        convinient_return=False,
        show_tqdm=True,
        exit_time=60
        )
    predictions = ArrayWithTime.from_list(result[0], drop_early_nans=True, squeeze_type='to_2d')

    Bubblewrap.compare_runs(bws=[bw], behavior_dicts=[{'predicted_behavior': predictions, 'true_behavior': behavioral_data}])
    if show:
        plt.show()

    # TODO: passing the regression results should be easier; maybe something like PredictionEvaluation?


if __name__ == '__main__':
    main()