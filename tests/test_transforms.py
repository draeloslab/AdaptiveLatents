import numpy as np
import adaptive_latents
import adaptive_latents.transforms as tr
from adaptive_latents.input_sources.timed_data_source import GeneratorDataSource, NumpyTimedDataSource
import pytest
import itertools


class TestTransformer:
    def test_can_handle_input_formats(self):
        for sources in [
            np.zeros((10,3)),
            np.zeros((5, 2, 3)),
            (np.zeros((2,3)) for _ in range(5)),
            [np.zeros((10, 3))],
            [(np.zeros((10, 3)), 1)],
            [(NumpyTimedDataSource(np.zeros((10, 3))), 0), (NumpyTimedDataSource(np.zeros((10, 3))), 0), (NumpyTimedDataSource(np.zeros((9, 3))), 1)],
        ]:
            t = tr.CenteringTransformer()
            t.offline_fit_transform(sources, convinient_return=False)


@pytest.mark.parametrize('transformer', [
    tr.CenteringTransformer(),
    tr.prosvd.TransformerProSVD(k=3, whiten=True),
    tr.jpca.TransformerSJPCA(),
    tr.ica.TransformerMMICA(),
    tr.transformer.Pipeline([
        tr.CenteringTransformer(),
        tr.prosvd.TransformerProSVD(k=4, whiten=False),
    ])
])
class TestPerTransformer:
    def test_can_ignore_nans(self, transformer: tr.TransformerMixin, rng):
        g = (rng.normal(size=(3,6)) * np.nan for _ in range(7))
        output = transformer.offline_fit_transform(g)

        g = (rng.normal(size=(3,6)) for _ in range(20))
        output = transformer.offline_fit_transform(g)
        assert (~np.isnan(output[-1])).all()

    # def test_can_handle_different_sizes(self):
    #     pass
    #
    # def test_can_reroute_stream(self):
    #     pass



class TestICA:
    def test_ica_runs(self):
        rng = np.random.default_rng()
        ica = tr.ica.mmICA()
        for _ in range(50):
            data = rng.laplace(size=(10, 10))
            ica.observe_new_batch(data)


class TestJPCA:
    def test_make_H(self, rng, dd=10):
        for d in range(2, dd):
            H = tr.jpca.sjPCA.make_H(d)
            for _ in range(100):
                k = rng.normal(size=int(d * (d-1) / 2))
                sksym = (H @ k).reshape((d, d))
                assert np.linalg.norm(np.real(np.linalg.eigvals(sksym))) < 1e-14

    def test_works_on_circular_data(self, rng):
        X, X_dot, true_variables = tr.jpca.generate_circle_embedded_in_high_d(rng, m=10_000, stddev=.01)

        jp = tr.jpca.sjPCA()
        X_realigned = jp.apply_to_data(X, show_tqdm=False)
        U = jp.last_U
        assert not np.allclose(U[:, :2], true_variables['C'])

        aligned_U, aligned_C = tr.jpca.align_column_spaces(U[:, :2], true_variables['C'])
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

            psvd1 = tr.proSVD.proSVD(k)
            psvd2 = tr.proSVD.proSVD(k)

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

            errors[0].append(tr.utils.column_space_distance(psvd1.Q, ideal_basis))
            errors[1].append(tr.utils.column_space_distance(psvd2.Q, ideal_basis))
        errors = np.array(errors)
        diff = errors[0] - errors[1]
        return (errors[0] - errors[1] > 0).mean(), diff.mean()

    def test_adding_colums_doesnt_hurt(self, rng):
        assert self.probabilistically_check_adding_channels_works(rng)[0] > .5

    def test_centering_works(self, rng):
        d = np.ones(10)
        d[0] = 2
        X = np.diag(d) @ rng.normal(size=(10, 500)) + 500
        psvd1 = tr.proSVD.proSVD(k=2, centering=False)
        psvd2 = tr.proSVD.proSVD(k=2, centering=True)

        psvd1.run_on(X)
        psvd2.run_on(X)

        _, s1, _ = np.linalg.svd(psvd1.B)
        _, s2, _ = np.linalg.svd(psvd2.B)

        assert abs(max(s1) / min(s1) - 2) > .5
        assert abs(max(s2) / min(s2) - 2) < .5


def test_utils_run(rng):
    # note I do not test correctness here
    A = rng.normal(size=(200, 10))
    t = np.arange(A.shape[0])

    A = adaptive_latents.transforms.utils.center_from_first_n(A)
    A = adaptive_latents.transforms.utils.zscore(A)
    A = adaptive_latents.transforms.utils.prosvd_data(input_arr=A, output_d=2, init_size=10, centering=True, _recalculate_cache_value=True)
    A = adaptive_latents.transforms.utils.prosvd_data(input_arr=A, output_d=2, init_size=10, centering=False, _recalculate_cache_value=True)
    A, t = adaptive_latents.transforms.utils.clip(A, t)

    A, Qs = adaptive_latents.transforms.utils.prosvd_data_with_Qs(input_arr=A, output_d=2, init_size=10)

    adaptive_latents.transforms.utils.bwrap_alphas(input_arr=A, bw_params=adaptive_latents.Bubblewrap.default_clock_parameters, _recalculate_cache_value=True)
    adaptive_latents.transforms.utils.bwrap_alphas_ahead(input_arr=A, bw_params=adaptive_latents.Bubblewrap.default_clock_parameters, _recalculate_cache_value=True)

    # this tests that y
    A = adaptive_latents.transforms.utils.prosvd_data(input_arr=A, output_d=2, init_size=10, centering=False, _recalculate_cache_value=True)


def test_can_use_args_in_cache(rng):
    A = rng.normal(size=(200, 10))
    A = adaptive_latents.transforms.utils.prosvd_data(input_arr=A, output_d=2, init_size=10, _recalculate_cache_value=True)
    A = adaptive_latents.transforms.utils.prosvd_data(input_arr=A, output_d=2, init_size=10, _recalculate_cache_value=False)
