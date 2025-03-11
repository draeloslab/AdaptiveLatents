import functools

import numpy as np
import pytest
from conftest import get_all_subclasses

from adaptive_latents import ArrayWithTime, CenteringTransformer, Pipeline, proSVD
from adaptive_latents.transformer import StreamingTransformer

DIM = 6

class TestStreamingTransformer:
    """
    This tests the code found in the StreamingTransformer class (as opposed to its subclasses).
    """
    transformer = CenteringTransformer()

    def test_streaming_run_on(self, valid_sources):
        for source in valid_sources:
            self.transformer.streaming_run_on(source)

    def test_offline_run_on(self, valid_sources):
        for source in valid_sources:
            self.transformer.offline_run_on(source, convinient_return=False)

    def test_trace_route(self):
        self.transformer.trace_route(stream=0)
        Pipeline([Pipeline([]), Pipeline([])]).trace_route(stream=0)

    @pytest.fixture
    def valid_sources(self):
        return [
            np.zeros((10, DIM)),
            np.zeros((5, 2, DIM)),
            (np.zeros((2, DIM)) for _ in range(5)),
            [((np.zeros((2, DIM)) for _ in range(5)), 0), ((np.zeros((2, DIM)) for _ in range(5)), 1)],
            [np.zeros((10, DIM))],
            [(np.zeros((10, DIM)), 1)],
            [(ArrayWithTime.from_notime(np.zeros((10, DIM))), 0), (ArrayWithTime.from_notime(np.zeros((10, DIM))), 0),
             (ArrayWithTime.from_notime(np.zeros((9, DIM))), 1)],
        ]




to_test = get_all_subclasses(StreamingTransformer)
to_test += [
    functools.partial(proSVD, k=DIM, whiten=True),
    functools.partial(Pipeline, [
        CenteringTransformer(),
        proSVD(k=DIM, whiten=False),
    ]),
    functools.partial(Pipeline, []),
]
@pytest.mark.parametrize('transformer_maker', to_test)
def test_all_transformers_are_api_compatible(transformer_maker, rng):
    t: StreamingTransformer = transformer_maker()
    t.test_if_api_compatible(constructor=transformer_maker, rng=rng, DIM=DIM)


# TODO: test if the appropriate logs are called for all iterations and all transformers (with mock functions)
# TODO: test if execution one-by-one or in a pipeline makes a difference
# TODO: what if there were a single object for multiple data streams?
