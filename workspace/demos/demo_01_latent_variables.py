import matplotlib.pyplot as plt

import adaptive_latents as al
from adaptive_latents import CenteringTransformer, KernelSmoother, Pipeline, proSVD, sjPCA

"""
Demo: Constructing latent variables
"""


def main(show_plots=True):
    d = al.datasets.Odoherty21Dataset()  # load a dataset
    neural_data = d.neural_data  # grab just the neural data

    # define a sequence of processing steps
    p = Pipeline([
        CenteringTransformer(),  # center the data
        KernelSmoother(),  # smooth the data
        proSVD(k=6),  # dimension reduce
        sjPCA()  # define a additional dimension reduction step
    ])

    latents = p.offline_run_on([neural_data])  # run the streaming pipeline over the neural data

    # plot the results
    fig, ax = plt.subplots()
    ax.scatter(latents[:, 0], latents[:, 1], s=1)
    if show_plots:
        plt.show()


if __name__ == '__main__':
    main()