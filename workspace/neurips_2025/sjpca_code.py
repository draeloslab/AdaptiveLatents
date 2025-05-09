from adaptive_latents import proSVD, sjPCA, Pipeline
import numpy as np
from adaptive_latents import ArrayWithTime
import matlab.engine


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
