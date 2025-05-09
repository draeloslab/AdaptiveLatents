from adaptive_latents import proSVD, mmICA, Pipeline, ArrayWithTime
from sklearn.decomposition import FastICA, PCA


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
