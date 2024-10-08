import numpy as np
import adaptive_latents as al
from adaptive_latents import NumpyTimedDataSource, CenteringTransformer, sjPCA, proSVD, mmICA, Pipeline, KernelSmoother, proPLS, Bubblewrap, RandomProjection, VanillaOnlineRegressor, ZScoringTransformer, Concatenator
from adaptive_latents.transformer import DecoupledTransformer, StreamingTransformer
from adaptive_latents.jpca import generate_circle_embedded_in_high_d
from adaptive_latents.utils import column_space_distance
import pytest
import copy
import itertools
import pickle
import matplotlib.pyplot as plt
import inspect

DIM = 6

streaming_only_transformers = [
    KernelSmoother,
    lambda: Bubblewrap(num=50)
]

decoupled_transformers = [
    CenteringTransformer,
    ZScoringTransformer,
    lambda: proSVD(k=DIM, whiten=True),
    lambda: proPLS(k=DIM),
    sjPCA,
    mmICA,
    RandomProjection,
    VanillaOnlineRegressor,
    lambda: Pipeline([
        CenteringTransformer(),
        proSVD(k=DIM, whiten=False),
    ]),
    lambda: Pipeline([]),
    Concatenator,
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
            [(NumpyTimedDataSource(np.zeros((10, DIM))), 0), (NumpyTimedDataSource(np.zeros((10, DIM))), 0), (NumpyTimedDataSource(np.zeros((9, DIM))), 1)],
        ]


@pytest.mark.parametrize('transformer_maker', all_transformers)
class TestPerStreamingTransformer:
    def test_can_fit_transform(self, transformer_maker, rng):
        transformer: StreamingTransformer = transformer_maker()
        for _ in range(5):
            for s in transformer.input_streams:
                transformer.partial_fit_transform(rng.normal(size=(10, DIM)), s)

    def test_can_save_and_rerun(self, transformer_maker, rng, tmp_path):
        transformer: StreamingTransformer = transformer_maker()

        for _ in range(5):
            for s in transformer.input_streams:
                transformer.partial_fit_transform(rng.normal(size=(10, DIM)), s)
        t2 = copy.deepcopy(transformer)

        temp_file = tmp_path/'streaming_transformer.pickle'
        with open(temp_file, 'bw') as f:
            pickle.dump(transformer, f)

        del transformer

        with open(temp_file, 'br') as f:
            transformer = pickle.load(f)

        for s in transformer.input_streams:
            x = rng.normal(size=(10, DIM))
            assert np.array_equal(transformer.partial_fit_transform(x, s),  t2.partial_fit_transform(x, s), equal_nan=True)

    def test_get_params_works(self, transformer_maker):
        transformer: StreamingTransformer = transformer_maker()
        p = {k:v for k, v in transformer.get_params().items() if len(k) and k[0] != "_"}
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
            A = rng.normal(size=(1,6))
            A_original = A.copy()
            f(A)
            assert np.all(A == A_original)

    def test_partial_fit_transform_decomposes_correctly(self, transformer_maker, rng):
        transformer: DecoupledTransformer = transformer_maker()

        for i in range(20):
            for stream in transformer.input_streams.keys():
                batch = rng.normal(size=(3,DIM))

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
                t2.partial_fit(batch, stream)

                assert np.array_equal(transformer.transform(batch), t2.transform(batch))

    def test_inverse_transform_works(self, transformer_maker, rng):
        transformer: DecoupledTransformer = transformer_maker()

        sources = self.make_sources(transformer, rng)
        transformer.offline_run_on(sources, convinient_return=False)
        try:
            output = transformer.inverse_transform(transformer.transform(rng.normal(size=(3,DIM))))
            assert output.shape == (3,DIM)
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

    def test_plots(self, rng):
        fig, ax = plt.subplots()

        X, _, true_variables = generate_circle_embedded_in_high_d(rng, m=100, stddev=.01)
        jp = sjPCA(log_level=1)
        jp.offline_run_on(X)

        jp.plot_U_stability(ax)
        jp.get_distance_from_subspace_over_time(true_variables['C'])





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

    def test_plots(self, rng):
        fig, ax = plt.subplots()

        X, _, true_variables = generate_circle_embedded_in_high_d(rng, m=100, stddev=.01)
        pro = proSVD(k=2, log_level=1)
        pro.offline_run_on(X)

        pro.plot_Q_stability(ax)
        pro.get_distance_from_subspace_over_time(true_variables['C'])

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

        pls = proPLS(k=3, log_level=1)

        pls.offline_run_on([X, Y])

        assert column_space_distance(pls.u, x_common_basis) < 0.155

        pls.get_distance_from_subspace_over_time(x_common_basis)

class TestBubblewrap:
    def test_plots(self, rng):
        bw = Bubblewrap(num=10, M=10, log_level=1)

        hmm = al.input_sources.hmm_simulation.HMM.gaussian_clock_hmm()
        states, observations = hmm.simulate_with_states(n_steps=50, rng=rng)

        bw.offline_run_on(observations)

        fig, axs = plt.subplots(nrows=4, ncols=4)
        axs = axs.flatten()

        i = -1
        bw.show_bubbles_2d(axs[(i:=i+1)])
        bw.show_alpha(axs[(i:=i+1)])
        bw.show_active_bubbles_2d(axs[(i:=i+1)])
        bw.show_active_bubbles_and_connections_2d(axs[(i:=i+1)], observations)
        bw.show_A(axs[(i:=i+1)])
        bw.show_nstep_pdf(ax=axs[(i:=i+1)], other_axis=axs[0], fig=fig, density=2)
        Bubblewrap.compare_runs([bw])

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
        c = Concatenator(input_streams={1:1, 2:2}, output_streams={1:0, 2:0})
        a = np.array([0,1,2]).reshape(-1,1)
        b = np.array([0,1,2]).reshape(-1,1)
        output = c.offline_run_on([(a,1), (b, 2)])
        assert (output == np.hstack((a,b))).all()

    def test_scales(self):
        c = Concatenator(input_streams={1:1, 2:2}, output_streams={1:0, 2:0}, stream_scaling_factors={1:1, 2:1})
        a = np.array([0,1,2]).reshape(-1,1)
        b = np.array([0,1,2]).reshape(-1,1)
        output = c.offline_run_on([(a,1), (b, 2)])
        assert (output == np.hstack((a,b))).all()

        c = Concatenator(input_streams={1:1, 2:2}, output_streams={1:0, 2:0}, stream_scaling_factors={1:2, 2:1})
        a = np.array([0,1,2]).reshape(-1,1)
        b = np.array([0,1,2]).reshape(-1,1)
        output = c.offline_run_on([(a,1), (b, 2)])
        assert (output == np.hstack((a*2,b))).all()




def test_miscellaneous_plots():
    fig, ax = plt.subplots()
    t = KernelSmoother()
    t.plot_impulse_response(ax)