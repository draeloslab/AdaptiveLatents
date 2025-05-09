import numpy as np
from adaptive_latents import ArrayWithTime, proSVD


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
