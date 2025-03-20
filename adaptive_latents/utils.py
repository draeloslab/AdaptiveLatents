import functools
import hashlib
import inspect
import json
import os
import pickle
import warnings
from collections import namedtuple

import numpy as np

from adaptive_latents.config import CONFIG


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.shape[0] > 1000:
                n_samples = 200
                row_samples = [round(x * (obj.shape[0] - 1)) for x in np.linspace(0, 1, n_samples)]
                obj = obj[row_samples]
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def make_hashable(x):
    return json.dumps(x, sort_keys=True, cls=NumpyEncoder).encode()


def make_hashable_and_hash(x):
    return int(hashlib.sha1(make_hashable(x)).hexdigest(), 16)


def save_to_cache(file, location=None, override_config_and_cache=False):
    location = location or CONFIG.cache_path

    if not CONFIG.attempt_to_cache and not override_config_and_cache:

        def decorator(original_function):
            @functools.wraps(original_function)
            def new_function(*args, _recalculate_cache_value=True, **kwargs):
                bound_args = inspect.signature(original_function).bind(*args, **kwargs)
                bound_args.apply_defaults()
                if not _recalculate_cache_value:
                    warnings.warn("don't try to cache when it's turned off in config")
                return original_function(**bound_args.arguments)

            return new_function

        return decorator

    cache_index_file = location / f"{file}_index.pickle"
    try:
        with open(cache_index_file, 'rb') as fhan:
            cache_index = pickle.load(fhan)
    except FileNotFoundError:
        cache_index = {}

    def decorator(original_function):
        @functools.wraps(original_function)
        def new_function(*args, _recalculate_cache_value=False, **kwargs):
            bound_args = inspect.signature(original_function).bind(*args, **kwargs)
            bound_args.apply_defaults()

            all_args = bound_args.arguments
            all_args_as_key = make_hashable_and_hash(all_args)

            if _recalculate_cache_value or all_args_as_key not in cache_index or not os.path.exists(location / cache_index[all_args_as_key]):
                result = original_function(**all_args)

                hstring = str(all_args_as_key)[-15:]
                cache_file = os.path.join(location, f"{file}_{hstring}.pickle")
                if CONFIG.verbose:
                    print(f"caching value in: {cache_file}")
                with CONFIG.open_with_parents(cache_file, "wb") as fhan:
                    pickle.dump(result, fhan)

                cache_index[all_args_as_key] = cache_file
                with CONFIG.open_with_parents(cache_index_file, 'bw') as fhan:
                    pickle.dump(cache_index, fhan)

            with open(location/ cache_index[all_args_as_key], 'rb') as fhan:
                if CONFIG.verbose:
                    # TODO: also log here
                    # TODO: have tests globally disable caching; you can recalculate, but that doesn't get inner caching
                    print(f"retreiving cache from: {cache_index[all_args_as_key]}")
                return pickle.load(fhan)

        return new_function

    return decorator



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


def check_same(v: np.ndarray):
    var = 'temp'
    import torch
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    s = f'/tmp/asdf_{var}'
    try:
        old_v = np.load(f"{s}.npy")
    except FileNotFoundError:
        old_v = None
    np.save(s, v)

    if old_v is not None:
        same = np.shape(v) == np.shape(old_v) and np.nanmax((v - old_v) ** 2) == 0
        print(f'{var}: {same}')
    else:
        print(f'{var}: NEW')
    return same


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


def is_orthonormal(Q, rows_too=False):
    o = np.allclose(Q.T @ Q, np.eye(Q.shape[1]))
    if rows_too:
        o = o and np.allclose(Q @ Q.t, np.eye(Q.shape[0]))
    return o


def column_space_distance(Q1, Q2, method='angles'):
    for Q in Q1, Q2:
        assert is_orthonormal(Q)

    if method == 'angles':
        return np.abs(principle_angles(Q1, Q2)).sum()
    elif method == 'aligned_diff':
        Q1_rotated, Q2 = align_column_spaces(Q1, Q2)
        return np.linalg.norm(Q1_rotated - Q2)
    else:
        raise ValueError()
