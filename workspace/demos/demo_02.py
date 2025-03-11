import matplotlib.pyplot as plt
import numpy as np

from adaptive_latents import ArrayWithTime, CenteringTransformer, KernelSmoother, Pipeline, proSVD, sjPCA

rng = np.random.default_rng(0)

"""
Demo: Using a new dataset
"""


def main(show=True):
    # use your code to load data here
    new_neural_data = rng.random(size=(1000,10))
    sample_times = np.linspace(0,1, new_neural_data.shape[0])

    # The library works best with ArrayWithTime instances; they're subclasses of numpy.ndarray that also track a t attribute.
    neural_data = ArrayWithTime(new_neural_data, sample_times)
    # You can use numpy arrays instead of ArrayWithTime, but they get converted to an ArrayWithTime where the
    # timesteps are np.arange(len(X)).

    p = Pipeline([
        CenteringTransformer(),
        KernelSmoother(),
        pro := proSVD(k=6, log_level=2),
        sjPCA()
    ])

    latents = p.offline_run_on([neural_data])

    fig, ax = plt.subplots()
    pro.plot_Q_stability(ax)  # note that the time goes from 0 to 1
    if show:
        plt.show()


if __name__ == '__main__':
    main()
