import numpy as np
from adaptive_latents.estimator import StreamingEstimator, DecoupledEstimator, pickle, copy, Predictor, ArrayWithTime
import warnings

def check_api_compatible(constructor, rng=None, DIM=None):
    check_streaming_estimator_api_compatible(constructor, rng, DIM)
    if isinstance(type(constructor()), DecoupledEstimator):
        check_decoupled_estimator_api_compatible(constructor, rng, DIM)

def check_streaming_estimator_api_compatible(constructor, rng=None, DIM=None):
    rng = rng or np.random.default_rng()
    DIM = DIM or 6

    StreamingEstimatorTests.test_get_params_works(constructor)
    StreamingEstimatorTests.test_can_fit_transform(constructor, rng, DIM)

    import pathlib
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        StreamingEstimatorTests.test_can_save_and_rerun(constructor, rng, tmp_path=pathlib.Path(tmp_dir), DIM=DIM)


def check_decoupled_estimator_api_compatible(constructor, rng=None, DIM=None):
    rng = rng or np.random.default_rng()
    DIM = DIM or 6

    DecoupledEstimatorTests.test_can_ignore_nans(constructor, rng)
    DecoupledEstimatorTests.test_original_matrix_unchanged(constructor, rng)
    DecoupledEstimatorTests.test_partial_fit_transform_decomposes_correctly(constructor, rng, DIM=DIM)
    DecoupledEstimatorTests.test_freezing_works_correctly(constructor, rng)
    DecoupledEstimatorTests.test_inverse_transform_works(constructor, rng, DIM=DIM)


class StreamingEstimatorTests:
    @staticmethod
    def test_can_fit_transform(constructor, rng, DIM=6):
        transformer: StreamingEstimator = constructor()
        for data, s in transformer.expected_data_streams(rng, DIM, cycles=5):
            transformer.step(data, s)

        # tests that the transformer can ignore data not in its input_sources
        # todo: make this Mock
        transformer.step(None, "test_that_this_doesn't go through")


    @staticmethod
    def test_can_save_and_rerun(constructor, rng, tmp_path, DIM=6):
        transformer: StreamingEstimator = constructor()

        for data, s in transformer.expected_data_streams(rng, DIM, cycles=5):
            transformer.step(data, s)
        t2 = copy.deepcopy(transformer)

        temp_file = tmp_path / 'streaming_transformer.pkl'
        with open(temp_file, 'bw') as f:
            pickle.dump(transformer, f)

        del transformer

        with open(temp_file, 'br') as f:
            transformer = pickle.load(f)

        for data, s in transformer.expected_data_streams(rng, DIM):
            a = transformer.step(data, s)
            b = t2.step(data, s)
            assert np.array_equal(a, b, equal_nan=True)


    @staticmethod
    def test_get_params_works(constructor):
        import inspect
        transformer: StreamingEstimator = constructor()
        p = {k: v for k, v in transformer.get_params().items() if len(k) and k[0] != "_"}
        type(transformer)(**p)

        base_signature = inspect.signature(transformer.base_algorithm)
        base_args = set(base_signature.parameters.keys())
        found_args = set(p.keys())
        assert base_args.issubset(found_args), 'you probably need to update get_params'

        found_signature = inspect.signature(type(transformer))
        for arg in base_args:
            if 'kwargs' in found_signature.parameters:
                continue
            base_default = base_signature.parameters[arg].default
            found_default = found_signature.parameters[arg].default
            assert (base_default is None) or (base_default is inspect.Parameter.empty) or base_default == found_default



class DecoupledEstimatorTests:
    @staticmethod
    def _make_sources(transformer, rng, expression=None, first_n_nan=0, length=20, DIM=6):
        import itertools
        if expression is None:
            expression = lambda: rng.normal(size=(3, DIM))

        batches = [expression() * (np.nan if i < first_n_nan else 1) for i in range(length)]
        return [tuple(x) for x in zip(itertools.repeat(batches), transformer.input_streams.keys())]

    @classmethod
    def test_can_ignore_nans(cls, constructor, rng):
        transformer = constructor()

        sources = cls._make_sources(transformer, rng, first_n_nan=7)
        transformer.offline_run_on(sources, convinient_return=False)

        sources = cls._make_sources(transformer, rng)
        output = transformer.offline_run_on(sources, convinient_return=False)

        for stream in output:
            assert (~np.isnan(output[stream][-1])).all()

    @classmethod
    def test_original_matrix_unchanged(cls, constructor, rng):
        transformer: DecoupledEstimator = constructor()

        sources = cls._make_sources(transformer, rng)
        transformer.offline_run_on(sources, convinient_return=False)

        for f in (transformer.partial_fit, transformer.transform):
            A = rng.normal(size=(1, 6))
            A_original = A.copy()
            f(A)
            assert np.all(A == A_original)

    @staticmethod
    def test_partial_fit_transform_decomposes_correctly(constructor, rng, DIM=6):
        transformer: DecoupledEstimator = constructor()

        for i in range(20):
            for stream in transformer.input_streams.keys():
                batch = rng.normal(size=(3, DIM))

                t1 = transformer
                t2 = copy.deepcopy(transformer)

                o1 = t1.step(batch, stream)

                t2.partial_fit(batch, stream)
                o2 = t2.transform(batch, stream)

                assert np.array_equal(o1, o2, equal_nan=True)

    @staticmethod
    def test_freezing_works_correctly(constructor, rng):
        transformer: DecoupledEstimator = constructor()

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

    @classmethod
    def test_inverse_transform_works(cls, constructor, rng, DIM=6):
        transformer: DecoupledEstimator = constructor()

        sources = cls._make_sources(transformer, rng)
        transformer.offline_run_on(sources, convinient_return=False)
        try:
            output = transformer.inverse_transform(transformer.transform(rng.normal(size=(3, DIM))))
            assert output.shape == (3, DIM)
        except NotImplementedError:
            pass

class PredictorTests:
    @classmethod
    def test_if_api_compatible(cls, constructor=None, rng=None, DIM=None):
        constructor, rng, DIM = super().test_if_api_compatible(constructor, rng, DIM)
        cls._test_checks_dt(constructor, rng, DIM)
        cls._test_output_t_is_origin_t(constructor, rng, DIM)


    @staticmethod
    def _test_output_t_is_origin_t(constructor, rng, DIM):
        predictor: Predictor = constructor()

        predictor.offline_run_on(rng.normal(size=(100, DIM)))

        output = predictor.step(ArrayWithTime([[1]], t=100), stream='dt_X')
        assert np.all(output.t == 100)


    @staticmethod
    def _test_checks_dt(constructor, rng, DIM):
        import pytest

        predictor: Predictor = constructor(check_dt=True)
        dt = 1 / np.pi
        predictor.step(ArrayWithTime(rng.normal(size=(1, DIM)), 0), stream='X')
        predictor.step(ArrayWithTime(rng.normal(size=(1, DIM)), 1 * dt), stream='X')

        assert np.isclose(predictor.dt, dt)

        predictor_backup = copy.deepcopy(predictor)

        # pytest_condition = pytest.raises(AssertionError)
        pytest_condition = pytest.warns(UserWarning, match='time steps for training are not consistent')

        with pytest_condition:
            warnings.warn('time steps for training are not consistent')

        predictor = copy.deepcopy(predictor_backup)
        with pytest.raises(AssertionError):
            predictor.step(ArrayWithTime(rng.normal(size=(1, DIM)), 1 * dt), stream='X')

        predictor = copy.deepcopy(predictor_backup)
        with pytest_condition:
            predictor.step(ArrayWithTime(rng.normal(size=(1, DIM)), 3 * dt), stream='X')

        predictor = copy.deepcopy(predictor_backup)
        predictor.step(ArrayWithTime(rng.normal(size=(1, DIM)), 2 * dt), stream='X')

        predictor = copy.deepcopy(predictor_backup)
        with pytest.raises(AssertionError):
            predictor.step(ArrayWithTime([[1]], 3 * dt), stream='dt_X')
