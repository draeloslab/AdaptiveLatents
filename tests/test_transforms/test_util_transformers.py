import numpy as np
from adaptive_latents import (
    KernelSmoother,
    ZScoringTransformer,
    Concatenator,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TestZScoringTransformer:
    def test_consistent(self, rng):
        X = rng.normal(size=(1000, 5)) * np.arange(5)
        z = ZScoringTransformer(freeze_after_init=False)
        z.offline_run_on(X)
        assert np.allclose(z.get_std(), np.std(X, axis=0), atol=0.01)

    # def test_unbiased(self, rng):
    #     pass


class TestConcatenator:
    def test_concatenates(self):
        c = Concatenator(input_streams={1: 1, 2: 2}, output_streams={1: 0, 2: 0})
        a = np.array([0, 1, 2]).reshape(-1, 1)
        b = np.array([0, 1, 2]).reshape(-1, 1)
        output = c.offline_run_on([(a, 1), (b, 2)])
        assert (output == np.hstack((a, b))).all()

    def test_scales(self):
        c = Concatenator(input_streams={1: 1, 2: 2}, output_streams={1: 0, 2: 0}, stream_scaling_factors={1: 1, 2: 1})
        a = np.array([0, 1, 2]).reshape(-1, 1)
        b = np.array([0, 1, 2]).reshape(-1, 1)
        output = c.offline_run_on([(a, 1), (b, 2)])
        assert (output == np.hstack((a, b))).all()

        c = Concatenator(input_streams={1: 1, 2: 2}, output_streams={1: 0, 2: 0}, stream_scaling_factors={1: 2, 2: 1})
        a = np.array([0, 1, 2]).reshape(-1, 1)
        b = np.array([0, 1, 2]).reshape(-1, 1)
        output = c.offline_run_on([(a, 1), (b, 2)])
        assert (output == np.hstack((a * 2, b))).all()


class TestKernelSmoother:
    def test_plots(self):
        fig, ax = plt.subplots()
        t = KernelSmoother()
        t.plot_impulse_response(ax)
