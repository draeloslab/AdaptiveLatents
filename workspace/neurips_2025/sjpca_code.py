from adaptive_latents import proSVD, sjPCA, Pipeline
import numpy as np
from adaptive_latents import ArrayWithTime

from adaptive_latents.jpca import generate_circle_embedded_in_high_d
from adaptive_latents.utils import principle_angles


def get_step_times(data, mid_d, low_d):
    p = Pipeline([proSVD(k=mid_d, log_level=0), sjPCA(log_level=0)], log_level=1)
    p.offline_run_on(data)
    step_times = ArrayWithTime.from_list(p.log["step_time"])
    return step_times


def get_projection_matrix_over_time(data, mid_d, low_d):
    p = Pipeline([proSVD(k=mid_d, log_level=2), sjPCA(log_level=2)])
    p.offline_run_on(data)
    Qs = ArrayWithTime.from_list(p.steps[0].log["Q"])
    Us = ArrayWithTime.from_list(p.steps[1].log["U"])
    Qs, Us = ArrayWithTime.align_indices(Qs, Us)
    return ArrayWithTime(Qs @ Us[:, :, :low_d], Qs.t)


def get_offline_projection_matrix(data, mid_d, low_d):
    import matlab.engine

    eng = matlab.engine.start_matlab()

    params = dict(
        meanSubtract=False,
        normalize=False,
        suppressBWrosettes=True,
        suppressHistograms=True,
        suppressText=True,
    )
    proj, summary = eng.jPCA({"A": data}, [], params, nargout=2)
    offline_U = np.array(summary["jPCs_highD"])
    return offline_U[:, :low_d]



def native_nearness_to_offline(mid_d, low_d, rng, T, dt, iterations_to_run, calculate_intra_run_errors=False):
    eng = matlab.engine.start_matlab()
    trajectories = []
    offline_errors = []
    for _ in range(iterations_to_run):
        X, _, true_variables = generate_circle_embedded_in_high_d(rng, m=int(T/dt), n=6, stddev=1)

        # matlab section
        params = dict(
            meanSubtract=False,
            normalize=False,
            suppressBWrosettes=True,
            suppressHistograms=True,
            suppressText=True,
        )
        proj, summary = eng.jPCA({'A':X}, [], params, nargout=2)
        offline_U = np.array(summary['jPCs_highD'])
        offline_error = np.abs(principle_angles(offline_U[:,:2], true_variables['C'])).sum()
        offline_errors.append(offline_error)

        # my section
        jp = sjPCA(log_level=2)
        jp.offline_run_on(ArrayWithTime(X, np.arange(X.shape[0]) * dt))
        distances = jp.get_distance_from_subspace_over_time(true_variables['C'])
        trajectories.append(distances[:,0])
    return offline_errors, trajectories, distances.t, 0, 'sjPCA'
