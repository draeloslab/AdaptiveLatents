import matplotlib.pyplot as plt

import adaptive_latents as al
from adaptive_latents import CenteringTransformer, Concatenator, KernelSmoother, Pipeline, Tee, proSVD, sjPCA

"""
Demo: Joint latent spaces
"""


def main(show_plots=True):
    d = al.datasets.Odoherty21Dataset()
    neural_data = d.neural_data
    behavioral_data = d.behavioral_data

    p = Pipeline([
        CenteringTransformer(),
        KernelSmoother(input_streams={0: 'X'}),  # this operates on data stream 0 (the neural data)
        KernelSmoother(input_streams={1: 'X'}),  # this operates on data stream 1 (the behavioral data)
        Concatenator(input_streams={0: 0, 1: 1}, output_streams={0: 0, 1: 0}),  # this concatenates stream 0 and stream 1 and returns the result to stream 0
        proSVD(k=6),
        tea := Tee(input_streams={0: 0}),
        sjPCA(),
    ])

    latents = p.offline_run_on([neural_data, behavioral_data])
    # `offline_run_on([(neural_data, 0), (behavioral_data, 1)])` syntax if you want to be more explicit that data stream 0
    # will carry the neural data and that data stream 1 will carry the behavioral data

    latents_before_sjpca = tea.convert_to_array()[0]

    fig, axs = plt.subplots(ncols=2)
    axs[0].scatter(latents_before_sjpca[:, 0], latents_before_sjpca[:, 1], s=1)
    axs[1].scatter(latents[:, 0], latents[:, 1], s=1)
    if show_plots:
        plt.show()


if __name__ == '__main__':
    main()