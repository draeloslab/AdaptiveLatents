import matplotlib.pyplot as plt

import adaptive_latents as al
from adaptive_latents import CenteringTransformer, KernelSmoother, Pipeline, proSVD, sjPCA

"""
Demo: Constructing latent variables
"""


def main(show=True):
    d = al.datasets.Odoherty21Dataset()
    neural_data = d.neural_data

    p = Pipeline([
        CenteringTransformer(),
        KernelSmoother(),
        proSVD(k=6),
        sjPCA()
    ])

    latents = p.offline_run_on([neural_data])

    fig, ax = plt.subplots()
    ax.scatter(latents[:, 0], latents[:, 1], s=1)
    if show:
        plt.show()


if __name__ == '__main__':
    main()