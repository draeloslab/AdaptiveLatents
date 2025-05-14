from adaptive_latents import ArrayWithTime, proSVD
import numpy as np
from adaptive_latents.utils import column_space_distance


from scipy.stats import special_ortho_group
def X_and_X_dot_from_data(X_all):
    """note: this is technically off-by-one for the way I normally think about it, but it's causal"""
    # todo: is this necessarily off-by-one?
    X_dot = np.diff(X_all, axis=0)
    X = X_all[1:]
    return X, X_dot
# from adaptive_latents.jpca import generate_circle_embedded_in_high_d
def generate_circle_embedded_in_high_d(rng, m=1000, n=4, stddev=1, shape=(10, 10)):
    t = np.linspace(0, (m / 10) * np.pi * 2, m + 1)
    circle = np.column_stack([np.cos(t), np.sin(t)]) @ np.diag(shape)
    C = special_ortho_group(dim=n, seed=rng).rvs()[:, :2]
    X_all = (circle @ C.T) + rng.normal(size=(m + 1, n)) * stddev
    X, X_dot = X_and_X_dot_from_data(X_all)
    return X, X_dot, dict(C=C)


def get_step_times(data, mid_d, low_d):
    pro = proSVD(k=mid_d, log_level=1)
    pro.offline_run_on(data)
    step_times = ArrayWithTime.from_list(pro.log["step_time"])
    return step_times


def get_projection_matrix_over_time(data, mid_d, low_d):
    pro = proSVD(k=mid_d, log_level=2)
    pro.offline_run_on(data)
    return ArrayWithTime.from_list(pro.log["Q"])[:, :, :low_d]


def get_offline_projection_matrix(data, mid_d, low_d):
    u, s, vh = np.linalg.svd(data, full_matrices=False)
    assert (np.diff(s) <= 0).all()
    return vh[:low_d].T

def native_nearness_to_offline(mid_d, low_d, rng, T, dt, iterations_to_run, calculate_intra_run_errors=False):
    trajectories = []
    offline_errors = []
    for _ in range(iterations_to_run):
        X, _, true_variables = generate_circle_embedded_in_high_d(rng, m=int(T/dt), n=8, stddev=1)
        pro = proSVD(k=4, log_level=2)
        X_with_time = ArrayWithTime(X, np.arange(X.shape[0]) * dt)
        pro.offline_run_on(X_with_time)
        Q_error = pro.get_distance_from_subspace_over_time(true_variables['C'])
        trajectories.append(Q_error)


        if calculate_intra_run_errors:
            intra_run_offline_errors = []
            for i in range(X.shape[0]):
                _, s, Vt = np.linalg.svd(X[:i])
                V = Vt[np.argsort(s)[::-1], :].T[:,:pro.k]
                offline_error = column_space_distance(V, true_variables['C'], method='angles')
                intra_run_offline_errors.append(offline_error)
            offline_errors.append(intra_run_offline_errors)
        else:
            _, s, Vt = np.linalg.svd(X)
            V = Vt[np.argsort(s)[::-1], :].T[:,:pro.k]
            offline_error = column_space_distance(V, true_variables['C'], method='angles')
            offline_errors.append(offline_error)

    return offline_errors, trajectories, trajectories[0].t, 2, 'proSVD', X_with_time.t




