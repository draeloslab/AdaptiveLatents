import numpy as np
import adaptive_latents as al
from adaptive_latents import NumpyTimedDataSource, CenteringTransformer, sjPCA, proSVD, mmICA, Pipeline, KernelSmoother, proPLS, Bubblewrap
from adaptive_latents.transformer import DecoupledTransformer, StreamingTransformer
from adaptive_latents.jpca import generate_circle_embedded_in_high_d
from adaptive_latents.utils import column_space_distance
import pytest
import copy
import itertools

streaming_only_transformers = [
    KernelSmoother,
    lambda: Bubblewrap(dim=6, num=50)
]

decoupled_transformers = [
    CenteringTransformer,
    lambda: proSVD(k=3, whiten=True),
    lambda: proPLS(k=3),
    sjPCA,
    mmICA,
    lambda: Pipeline([
        CenteringTransformer(),
        proSVD(k=3, whiten=False),
    ]),
    lambda: Pipeline([])
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

    @pytest.fixture
    def valid_sources(self):
        return [
            np.zeros((10, 3)),
            np.zeros((5, 2, 3)),
            (np.zeros((2,3)) for _ in range(5)),
            [((np.zeros((2,3)) for _ in range(5)), 0), ((np.zeros((2,3)) for _ in range(5)), 1)],
            [np.zeros((10, 3))],
            [(np.zeros((10, 3)), 1)],
            [(NumpyTimedDataSource(np.zeros((10, 3))), 0), (NumpyTimedDataSource(np.zeros((10, 3))), 0), (NumpyTimedDataSource(np.zeros((9, 3))), 1)],
        ]


@pytest.mark.parametrize('transformer_maker', all_transformers)
class TestPerStreamingTransformer:
    def test_can_fit_transform(self, transformer_maker):
        transformer: StreamingTransformer = transformer_maker()

        transformer.partial_fit_transform(np.zeros((10,3)))


@pytest.mark.parametrize('transformer_maker', decoupled_transformers)
class TestPerDecoupledTransformer:
    @staticmethod
    def make_sources(transformer, rng, expression=None, first_n_nan=0, length=20):
        if expression is None:
            expression = lambda: rng.normal(size=(3, 6))

        batches = (expression() * (np.nan if i < first_n_nan else 1) for i in range(length))
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
            A = rng.normal(size=(1,6))
            A_original = A.copy()
            f(A)
            assert np.all(A == A_original)

    def test_partial_fit_transform_decomposes_correctly(self, transformer_maker, rng):
        transformer: DecoupledTransformer = transformer_maker()

        for i in range(20):
            for stream in transformer.input_streams.keys():
                batch = rng.normal(size=(3,6))

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
                batch = rng.normal(size=(2,6))
                transformer.partial_fit(batch, stream)
        t2 = copy.deepcopy(transformer)

        transformer.freeze(True)
        for i in range(10):
            for stream in transformer.input_streams.keys():
                batch = rng.normal(size=(2,6))
                transformer.partial_fit(batch, stream)
                assert np.array_equal(transformer.transform(batch), t2.transform(batch))

        transformer.freeze(False)
        for i in range(10):
            for stream in transformer.input_streams.keys():
                batch = rng.normal(size=(2, 6))
                transformer.partial_fit(batch, stream)

                if i > 0:  # some algorithms need a sample from each stream to update
                    t2_result = t2.transform(batch, stream)
                    assert np.array_equal(batch, t2_result) == np.array_equal(transformer.transform(batch, stream), t2_result)

    def test_inverse_transform_works(self, transformer_maker, rng):
        transformer: DecoupledTransformer = transformer_maker()

        sources = self.make_sources(transformer, rng)
        transformer.offline_run_on(sources, convinient_return=False)
        try:
            output = transformer.inverse_transform(transformer.transform(rng.normal(size=(3,6))))
            assert output.shape == (3,6)
        except NotImplementedError:
            pass

    # TODO: test if wrapping generator data sources works correctly
    # TODO: test if the appropriate logs are called for all iterations and all transformers (with mock functions)
    # TODO: test if execution one-by-one or in a pipeline makes a difference
    # TODO: what if there were a single object for multiple data streams?
    # TODO: try all of the input streams programatically
    # TODO: test if time is passed through appropriately





class TestICA:
    def test_ica_runs(self):
        rng = np.random.default_rng()
        ica = mmICA()
        for _ in range(50):
            data = rng.laplace(size=(10, 10))
            ica.observe_new_batch(data)


class TestJPCA:
    def test_make_H(self, rng, dd=10):
        for d in range(2, dd):
            H = sjPCA.make_H(d)
            for _ in range(100):
                k = rng.normal(size=int(d * (d-1) / 2))
                sksym = (H @ k).reshape((d, d))
                assert np.linalg.norm(np.real(np.linalg.eigvals(sksym))) < 1e-14

    def test_works_on_circular_data(self, rng):
        X, X_dot, true_variables = generate_circle_embedded_in_high_d(rng, m=10_000, stddev=.01)

        jp = sjPCA()
        X_realigned = jp.offline_run_on(X, convinient_return=True)
        U = jp.last_U
        assert not np.allclose(U[:, :2], true_variables['C'])

        aligned_U, aligned_C = al.utils.align_column_spaces(U[:, :2], true_variables['C'])
        assert np.allclose(aligned_U, aligned_C, atol=1e-4)


class TestProSVD:
    def probabilistically_check_adding_channels_works(self, rng, n_samples=50, n1=4, n2=10, k=2):
        errors = [[], []]
        for _ in range(n_samples):
            d = np.ones(n2)
            d[0] = 1.3
            d[-1] = 10
            d = np.diag(d)
            sub_d = d[:-(n2 - n1), :-(n2 - n1)] if n1 != n2 else d

            data = [
                sub_d @ rng.normal(size=(n1, n1)),
                sub_d @ rng.normal(size=(n1, 5)),
                d @ rng.normal(size=(n2, n2)),
                d @ rng.normal(size=(n2, 5)),
            ]

            psvd1 = proSVD(k=k)
            psvd2 = proSVD(k=k)

            psvd1.initialize(data[2])
            for i in np.arange(data[3].shape[1]):
                psvd1.updateSVD(data[3][:, i:i + 1])

            psvd2.initialize(data[0])
            for i in np.arange(data[1].shape[1]):
                psvd2.updateSVD(data[1][:, i:i + 1])

            psvd2.add_new_input_channels(n2 - n1)
            for j in [2, 3]:
                for i in np.arange(data[j].shape[1]):
                    psvd2.updateSVD(data[j][:, i:i + 1])

            ideal_basis = np.zeros((n2, 2))
            ideal_basis[0, 0] = 1
            ideal_basis[-1, 1] = 1

            errors[0].append(column_space_distance(psvd1.Q, ideal_basis, method='aligned_diff'))
            errors[1].append(column_space_distance(psvd2.Q, ideal_basis, method='aligned_diff'))
        errors = np.array(errors)
        diff = errors[0] - errors[1]
        return (errors[0] - errors[1] > 0).mean(), diff.mean()

    def test_adding_colums_doesnt_hurt(self, rng):
        assert self.probabilistically_check_adding_channels_works(rng)[0] > .5

    def test_can_find_subspace(self, rng):
        X, _, true_variables = generate_circle_embedded_in_high_d(rng, m=500, n=8, stddev=1)
        pro = proSVD(k=3)
        pro.offline_run_on(X)

        assert column_space_distance(pro.Q, true_variables['C']) < 0.05

    # TODO:
    # def test_n_samples_works_with_decay_alpha(self):
    #     assert False


class TestProPLS:
    def test_reconstruction(self, rng):
        base_d = 5
        high_d = (10, 9)
        n_points = 150

        X = rng.normal(size=(n_points, base_d))
        Y = rng.normal(size=(n_points, base_d))
        X_original = np.hstack([X, np.zeros((n_points, high_d[0] - base_d))])
        Y_original = np.hstack([Y, np.zeros((n_points, high_d[1] - base_d))])

        X = NumpyTimedDataSource(X_original.reshape(30, 5, -1))
        Y = NumpyTimedDataSource(Y_original.reshape(30, 5, -1))

        pls = proPLS(k=5)
        pls.offline_run_on([X, Y])

        assert np.allclose(X_original.T @ Y_original, pls.get_cross_covariance())

    def test_finds_subspace(self, rng):
        high_d = (10, 9)
        n_points = 2000
        common_d = 2

        X = rng.normal(size=(n_points, high_d[0]))
        Y = rng.normal(size=(n_points, high_d[1]))

        snr = 100
        common = rng.normal(size=(n_points, common_d))
        Y[:,:common_d] = (snr * common + rng.normal(size=(n_points, common_d)))/np.sqrt(1 + snr**2)
        X[:,:common_d] = (snr * common + rng.normal(size=(n_points, common_d)))/np.sqrt(1 + snr**2)

        x_common_basis = np.eye(high_d[0])[:,:common_d]

        pls = proPLS(k=3)

        pls.offline_run_on([X, Y])

        assert column_space_distance(pls.u, x_common_basis) < 0.155


def test_utils_run(rng):
    # note I do not test correctness here
    A = rng.normal(size=(200, 10))
    t = np.arange(A.shape[0])

    # A = al.utils.center_from_first_n(A)
    A = al.utils.zscore(A)
    # A = al.utils.prosvd_data(input_arr=A, output_d=2, init_size=10, centering=True, _recalculate_cache_value=True)
    # A = al.utils.prosvd_data(input_arr=A, output_d=2, init_size=10, centering=False, _recalculate_cache_value=True)
    A, t = al.utils.clip(A, t)

    # A, Qs = al.utils.prosvd_data_with_Qs(input_arr=A, output_d=2, init_size=10)

    # al.utils.bwrap_alphas(input_arr=A, bw_params=al.Bubblewrap.default_clock_parameters, _recalculate_cache_value=True)
    # al.utils.bwrap_alphas_ahead(input_arr=A, bw_params=al.Bubblewrap.default_clock_parameters, _recalculate_cache_value=True)

    # this tests that y
    # A = al.utils.prosvd_data(input_arr=A, output_d=2, init_size=10, centering=False, _recalculate_cache_value=True)


# def test_can_use_args_in_cache(rng):
#     A = rng.normal(size=(200, 10))
#     A = adaptive_latents.transforms.utils.prosvd_data(input_arr=A, output_d=2, init_size=10, _recalculate_cache_value=True)
#     A = adaptive_latents.transforms.utils.prosvd_data(input_arr=A, output_d=2, init_size=10, _recalculate_cache_value=False)
