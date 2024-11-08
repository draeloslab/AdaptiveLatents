import numpy as np
from collections import namedtuple


def clip(*args, maxlen=float("inf")):
    """take a variable number of arguments and trim them to be the same length

    The logic behind this function is that lots of the time arrays become misaligned because some initialiation cut off the early values of one of the arrays.
    This function hopes to re-align variable-length arrays by only keeping the last N values.
    It also trims off NaN's in the beginning of an array as if they were missing values.

    inputs:
        *args: a set of iterables
        maxlen: a maximum length to trim them all down to (defaults to the shortest of the lengths of the trimmed iterables)

    outputs:
         clipped_arrays: the arrays passed in as `*args`, but shortened
    """
    l = min([len(a) for a in args])
    l = int(min(maxlen, l))
    args = [a[-l:] for a in args]

    m = 0
    for arg in args:
        fin = np.isfinite(arg)
        if len(fin.shape) > 1:
            assert len(fin.shape) == 2
            fin = np.all(fin, axis=1)
        m = max(m, np.nonzero(fin)[0][0])

    clipped_arrays = [a[m:] for a in args]
    return clipped_arrays


def resample_matched_timeseries(old_timeseries, old_sample_times, new_sample_times,):
    good_samples = ~np.any(np.isnan(old_timeseries), axis=1)
    resampled_behavior = np.zeros((new_sample_times.shape[0], old_timeseries.shape[1]))
    for c in range(resampled_behavior.shape[1]):
        resampled_behavior[:, c] = np.interp(new_sample_times, old_sample_times[good_samples], old_timeseries[good_samples, c])
    return resampled_behavior


def evaluate_regression(estimate, estimate_t,  target, target_t):
    t = estimate_t
    targets = resample_matched_timeseries(
        target,
        target_t,
        estimate_t
    )

    test_s = t > (t[0] + t[-1]) / 2

    correlations = np.array([np.corrcoef(estimate[test_s, i], targets[test_s, i])[0, 1] for i in range(estimate.shape[1])])
    nrmse_s = np.sqrt(((estimate[test_s] - targets[test_s]) ** 2).mean(axis=0)) / targets[test_s].std(axis=0)

    EvalResult = namedtuple('EvalResult', ['corr', 'nrmse'])
    return EvalResult(correlations, nrmse_s)


def align_column_spaces(A, B):
    # https://simonensemble.github.io/posts/2018-10-27-orthogonal-procrustes/
    # R = argmin(lambda omega: norm(omega @ A - B))
    A, B = A.T, B.T
    C = A @ B.T
    u, s, vh = np.linalg.svd(C)
    R = vh.T @ u.T
    return (R @ A).T, (B).T


def principle_angles(Q1, Q2):
    _, s, _ = np.linalg.svd(Q1.T @ Q2)
    return np.arccos(s)

def column_space_distance(Q1, Q2, method='angles'):
    # for Q in Q1, Q2:
    #     assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]))

    if method == 'angles':
        return np.abs(principle_angles(Q1, Q2)).sum()
    elif method == 'aligned_diff':
        Q1_rotated, Q2 = align_column_spaces(Q1, Q2)
        return np.linalg.norm(Q1_rotated - Q2)
    else:
        raise ValueError()
