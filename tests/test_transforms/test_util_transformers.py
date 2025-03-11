import matplotlib
import numpy as np

from adaptive_latents import Concatenator, KernelSmoother

matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
