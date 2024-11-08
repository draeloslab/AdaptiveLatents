import numpy as np
from adaptive_latents import (
    KernelSmoother,
    ZScoringTransformer,
    Concatenator,
    Pipeline,
    CenteringTransformer,
)
import adaptive_latents
from joblib import Memory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest


class TestZScoringTransformer:
    def test_consistent(self, rng):
        X = rng.normal(size=(1000, 5)) * np.arange(5)
        z = ZScoringTransformer(freeze_after_init=False)
        z.offline_run_on(X)
        assert np.allclose(z.get_std(), np.std(X, axis=0), atol=0.01)

class TestPipeline:
    # @pytest.mark.parametrize("pseudobatch,cached", [ (0, 0), (0, 1), (1, 0), (1, 1) ])
    # def test_caches(self, tmp_path, rng, mocker, pseudobatch, cached):
    #     adaptive_latents.CONFIG.caching = Memory(location=tmp_path, verbose=False)
    #     adaptive_latents.CONFIG.attempt_to_cache = True
    #
    #     X = rng.normal(size=(100, 5))
    #
    #     p1 = Pipeline([CenteringTransformer(), CenteringTransformer()], pseudobatch=pseudobatch)
    #     o1 = p1.offline_run_on(X, cached=cached)
    #
    #     p2 = Pipeline([CenteringTransformer(), CenteringTransformer()], pseudobatch=pseudobatch)
    #     pipeline_pft = p2.partial_fit_transform = mocker.Mock(name='partial_fit_transform', side_effect=p2.partial_fit_transform)
    #     inner_pft = p2.steps[0].partial_fit_transform = mocker.Mock(name='partial_fit_transform', side_effect=p2.steps[0].partial_fit_transform)
    #     o2 = p2.offline_run_on(X, cached=cached)
    #
    #     if cached:
    #         assert (not inner_pft.called) and (not pipeline_pft.called)
    #     elif pseudobatch:
    #         assert inner_pft.called and (not pipeline_pft.called)
    #     else:
    #         assert inner_pft.called and pipeline_pft.called

    def test_pseudobatch_could_equal_streaming(self, rng):
        X = rng.normal(size=(100, 5))

        p1 = Pipeline([CenteringTransformer(), CenteringTransformer()])
        p2 = Pipeline([CenteringTransformer(), CenteringTransformer()])

        o1 = p1.offline_run_on(X)

        o2 = {0:X}
        for step in p2.steps:
            o2 = step.offline_run_on(o2)

        assert (o1 == o2).all()

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
