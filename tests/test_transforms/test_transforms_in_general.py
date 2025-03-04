import numpy as np
from adaptive_latents import (
    CenteringTransformer,
    sjPCA,
    proSVD,
    mmICA,
    Pipeline,
    KernelSmoother,
    proPLS,
    Bubblewrap,
    RandomProjection,
    VanillaOnlineRegressor,
    ZScoringTransformer,
    Concatenator,
    ArrayWithTime
)
from adaptive_latents.transformer import DecoupledTransformer, StreamingTransformer
import pytest
import copy
import itertools
import pickle
import inspect
import functools

DIM = 6

streaming_only_transformers = [
    KernelSmoother,
    functools.partial(Bubblewrap, num=50),
    Concatenator,
]

decoupled_transformers = [
    CenteringTransformer,
    functools.partial(ZScoringTransformer, init_size=10),
    functools.partial(proSVD, k=DIM, whiten=True),
    functools.partial(proPLS, k=DIM),
    sjPCA,
    mmICA,
    RandomProjection,
    VanillaOnlineRegressor,
    functools.partial(Pipeline, [
        CenteringTransformer(),
        proSVD(k=DIM, whiten=False),
    ]),
    functools.partial(Pipeline, []),
]

all_transformers = streaming_only_transformers + decoupled_transformers


class TestStreamingTransformer:
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


@pytest.mark.parametrize('transformer_maker', all_transformers)
class TestPerStreamingTransformer:
    def test_can_fit_transform(self, transformer_maker, rng):
        transformer: StreamingTransformer = transformer_maker()
        for _ in range(5):
            for data, s in transformer.expected_data_streams(rng, DIM):
                transformer.partial_fit_transform(data, s)

    def test_can_save_and_rerun(self, transformer_maker, rng, tmp_path):
        transformer: StreamingTransformer = transformer_maker()

        for _ in range(5):
            for data, s in transformer.expected_data_streams(rng, DIM):
                transformer.partial_fit_transform(data, s)
        t2 = copy.deepcopy(transformer)

        temp_file = tmp_path / 'streaming_transformer.pickle'
        with open(temp_file, 'bw') as f:
            pickle.dump(transformer, f)

        del transformer

        with open(temp_file, 'br') as f:
            transformer = pickle.load(f)

        for data, s in transformer.expected_data_streams(rng, DIM):
            assert np.array_equal(transformer.partial_fit_transform(data, s), t2.partial_fit_transform(data, s),
                                  equal_nan=True)

    def test_get_params_works(self, transformer_maker):
        transformer: StreamingTransformer = transformer_maker()
        p = {k: v for k, v in transformer.get_params().items() if len(k) and k[0] != "_"}
        type(transformer)(**p)

        base_algorithm = transformer.base_algorithm
        base_args = set(inspect.signature(base_algorithm).parameters.keys()) - {'args', 'kwargs'}
        found_args = set(p.keys()) - {'args', 'kwargs'}
        assert base_args.issubset(found_args)


@pytest.mark.parametrize('transformer_maker', decoupled_transformers)
class TestPerDecoupledTransformer:
    @staticmethod
    def make_sources(transformer, rng, expression=None, first_n_nan=0, length=20):
        if expression is None:
            expression = lambda: rng.normal(size=(3, DIM))

        batches = [expression() * (np.nan if i < first_n_nan else 1) for i in range(length)]
        return list(zip(itertools.repeat(batches), transformer.input_streams.keys()))

    def test_can_ignore_nans(self, transformer_maker, rng):
        transformer: DecoupledTransformer = transformer_maker()

        sources = self.make_sources(transformer, rng, first_n_nan=7)
        transformer.offline_run_on(sources, convinient_return=False)

        sources = self.make_sources(transformer, rng)
        output = transformer.offline_run_on(sources, convinient_return=False)

        for stream in output:
            assert (~np.isnan(output[stream][-1])).all()

    def test_original_matrix_unchanged(self, transformer_maker, rng):
        transformer: DecoupledTransformer = transformer_maker()

        sources = self.make_sources(transformer, rng)
        transformer.offline_run_on(sources, convinient_return=False)

        for f in (transformer.partial_fit, transformer.transform):
            A = rng.normal(size=(1, 6))
            A_original = A.copy()
            f(A)
            assert np.all(A == A_original)

    def test_partial_fit_transform_decomposes_correctly(self, transformer_maker, rng):
        transformer: DecoupledTransformer = transformer_maker()

        for i in range(20):
            for stream in transformer.input_streams.keys():
                batch = rng.normal(size=(3, DIM))

                t1 = transformer
                t2 = copy.deepcopy(transformer)

                o1 = t1.partial_fit_transform(batch, stream)

                t2.partial_fit(batch, stream)
                o2 = t2.transform(batch, stream)

                assert np.array_equal(o1, o2, equal_nan=True)

    def test_freezing_works_correctly(self, transformer_maker, rng):
        transformer: DecoupledTransformer = transformer_maker()

        transformer.freeze(False)
        for i in range(10):
            for stream in transformer.input_streams.keys():
                batch = rng.normal(size=(2, 6))
                transformer.partial_fit(batch, stream)
        t2 = copy.deepcopy(transformer)

        transformer.freeze(True)
        for i in range(10):
            for stream in transformer.input_streams.keys():
                batch = rng.normal(size=(2, 6))
                transformer.partial_fit(batch, stream)
                assert np.array_equal(transformer.transform(batch), t2.transform(batch))

        transformer.freeze(False)
        for i in range(10):
            for stream in transformer.input_streams.keys():
                batch = rng.normal(size=(2, 6))
                transformer.partial_fit(batch, stream)
                t2.partial_fit(batch, stream)

                assert np.array_equal(transformer.transform(batch), t2.transform(batch))

    def test_inverse_transform_works(self, transformer_maker, rng):
        transformer: DecoupledTransformer = transformer_maker()

        sources = self.make_sources(transformer, rng)
        transformer.offline_run_on(sources, convinient_return=False)
        try:
            output = transformer.inverse_transform(transformer.transform(rng.normal(size=(3, DIM))))
            assert output.shape == (3, DIM)
        except NotImplementedError:
            pass

    # TODO: test if wrapping generator data sources works correctly
    # TODO: test if the appropriate logs are called for all iterations and all transformers (with mock functions)
    # TODO: test if execution one-by-one or in a pipeline makes a difference
    # TODO: what if there were a single object for multiple data streams?
    # TODO: try all of the input streams programatically
    # TODO: test if time is passed through appropriately
