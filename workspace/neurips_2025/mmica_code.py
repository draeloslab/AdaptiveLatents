from adaptive_latents import proSVD, mmICA, Pipeline, ArrayWithTime
from sklearn.decomposition import FastICA, PCA
import numpy as np
from picard import permute


def get_step_times(data, mid_d, low_d):
    p = Pipeline([proSVD(k=mid_d, log_level=0), mmICA(log_level=0)], log_level=1)
    p.offline_run_on(data)
    step_times = ArrayWithTime.from_list(p.log["step_time"])
    return step_times


def get_projection_matrix_over_time(data, mid_d, low_d):
    p = Pipeline([proSVD(k=mid_d, log_level=2), mmICA(log_level=2)])
    p.offline_run_on(data)
    Qs = ArrayWithTime.from_list(p.steps[0].log["Q"])
    Ws = ArrayWithTime.from_list(p.steps[1].log["W"])
    Qs, Ws = ArrayWithTime.align_indices(Qs, Ws)
    Ws = Ws.transpose((0, 2, 1))
    Ws = Ws[:, :, :low_d]
    return ArrayWithTime(Qs @ Ws, Qs.t)


def get_offline_projection_matrix(data, mid_d, low_d):
    pca = PCA(n_components=mid_d)
    data = pca.fit_transform(data)
    W = (
        FastICA(max_iter=5000).fit(data).components_.T
    )  # TODO: check that it's supposed to be transposed
    return pca.components_.T @ W[:, :low_d]


def native_nearness_to_offline(mid_d, low_d, rng, T, dt, iterations_to_run, calculate_intra_run_errors=False):
    trajectories = []
    offline_errors = []
    for _ in range(iterations_to_run):
        n = 6
        m = int((T/dt)//n) # number of blocks
        X = rng.laplace(size=(m, n, n))

        ica = mmICA(alpha=.5, maxiter_cg=20, tol=1e-20, log_level=2)
        input_data = ArrayWithTime(X, np.arange(m) * n * dt)
        ica.offline_run_on(input_data)
        ts = []
        errors = []
        for W in ica.log['W']:
            error = permute(W) - np.eye(W.shape[0])
            errors.append(ArrayWithTime(np.linalg.norm(error), W.t))
        errors = ArrayWithTime.from_list(errors, squeeze_type='squeeze')
        trajectories.append(errors)


        # intra_run_offline_errors = []
        # for i in range(X.shape[0]):
        #     pass
        W = FastICA(max_iter=5000).fit(X.transpose([0,2,1]).reshape(-1,n)).components_
        offline_error = np.linalg.norm(permute(W) - np.eye(W.shape[0]))
        offline_errors.append(offline_error)


    return offline_errors, trajectories, errors.t, 1, 'mmICA'
