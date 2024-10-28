import numpy as np
import adaptive_latents as al
from adaptive_latents import (
    NumpyTimedDataSource,
    sjPCA,
    proSVD,
    mmICA,
    proPLS,
)
from adaptive_latents.jpca import generate_circle_embedded_in_high_d
from adaptive_latents.utils import column_space_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
                k = rng.normal(size=int(d * (d - 1) / 2))
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
        Y[:, :common_d] = (snr * common + rng.normal(size=(n_points, common_d))) / np.sqrt(1 + snr ** 2)
        X[:, :common_d] = (snr * common + rng.normal(size=(n_points, common_d))) / np.sqrt(1 + snr ** 2)

        x_common_basis = np.eye(high_d[0])[:, :common_d]

        pls = proPLS(k=3, log_level=1)

        pls.offline_run_on([X, Y])

        assert column_space_distance(pls.u, x_common_basis) < 0.155

        pls.get_distance_from_subspace_over_time(x_common_basis)
