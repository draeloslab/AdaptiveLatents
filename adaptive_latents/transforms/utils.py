import numpy as np
import os
import pickle
import json
from adaptive_latents.transforms import proSVD
from tqdm import tqdm
import hashlib
from adaptive_latents.config import CONFIG
import adaptive_latents
import inspect
import warnings
import functools


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


def save_to_cache(file, location=CONFIG["cache_path"]):
    if not CONFIG["attempt_to_cache"]:

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

    if not os.path.exists(location):
        os.makedirs(location)
    cache_index_file = os.path.join(location, f"{file}_index.pickle")
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
                if CONFIG["verbose"]:
                    print(f"caching value in: {cache_file}")
                with open(cache_file, "wb") as fhan:
                    pickle.dump(result, fhan)

                cache_index[all_args_as_key] = cache_file
                with open(cache_index_file, 'bw') as fhan:
                    pickle.dump(cache_index, fhan)

            with open(os.path.join(location, cache_index[all_args_as_key]), 'rb') as fhan:
                if CONFIG["verbose"]:
                    # TODO: also log here
                    # TODO: have tests globally disable caching; you can recalculate, but that doesn't get inner caching
                    print(f"retreiving cache from: {cache_index[all_args_as_key]}")
                return pickle.load(fhan)

        return new_function

    return decorator


@save_to_cache("prosvd_data")
def prosvd_data(input_arr, output_d, init_size, centering=True):
    # todo: rename this and the sjPCA version to apply_and_cache?
    pro = proSVD(k=output_d, centering=centering)
    pro.initialize(input_arr[:init_size].T)

    output = []
    for i in tqdm(range(init_size, len(input_arr))):
        obs = input_arr[i:i + 1, :]
        if not np.all(~np.isnan(obs)):
            output.append(np.nan * output[-1])
            continue

        output.append(pro.update_and_project(obs.T))
    return np.array(output).reshape((-1, output_d))


def prosvd_data_with_Qs(input_arr, output_d, init_size, pro_arguments=None):
    # todo: combine this with prosvd_data
    pro_arguments = pro_arguments or {}
    pro = proSVD(k=output_d, **pro_arguments)
    pro.initialize(input_arr[:init_size].T)

    output = []
    old_Qs = []
    for i in range(init_size, len(input_arr)):
        obs = input_arr[i:i + 1, :]
        old_Qs.append(np.array(pro.Q))
        output.append(pro.update_and_project(obs.T))
    return np.array(output).reshape((-1, output_d)), np.array(old_Qs)


def zscore(input_arr, init_size=6, clip_level=15):
    mean = 0
    m2 = 1e-4
    output = []
    count = 0
    for i, x in enumerate(tqdm(input_arr)):
        if np.any(np.isnan(x)):
            output.append(x * np.nan)
            continue

        if count >= init_size:
            std = np.sqrt(m2 / (count-1))
            output.append((x-mean) / std)

        delta = x - mean
        mean += delta / (count+1)
        m2 += delta * (x-mean)
        count += 1
    output = np.array(output)

    if clip is not None:
        output[output > clip_level] = clip_level
        output[output < -clip_level] = -clip_level
    return output


# todo: some rank-version of zscore?


@save_to_cache("bwrap_alphas")
def bwrap_alphas(input_arr, bw_params):
    alphas = []
    bw = adaptive_latents.Bubblewrap(dim=input_arr.shape[1], **bw_params)
    for step in range(len(input_arr)):
        bw.observe(input_arr[step])

        if step < bw.M:
            pass
        elif step == bw.M:
            bw.init_nodes()
            bw.e_step()
            bw.grad_Q()
        else:
            bw.e_step()
            bw.grad_Q()

            alphas.append(bw.alpha)
    return np.array(alphas)


@save_to_cache("bwrap_alphas_ahead")
def bwrap_alphas_ahead(input_arr, bw_params, nsteps=(1,)):
    returns = {x: [] for x in nsteps}
    bw = adaptive_latents.Bubblewrap(dim=input_arr.shape[1], **bw_params)
    for step in tqdm(range(len(input_arr))):
        bw.observe(input_arr[step])

        if step < bw.M:
            pass
        elif step == bw.M:
            bw.init_nodes()
            bw.e_step()
            bw.grad_Q()
        else:
            bw.e_step()
            bw.grad_Q()

            for step in nsteps:
                returns[step].append(bw.alpha @ np.linalg.matrix_power(bw.A, step))
    returns = {x: np.array(returns[x]) for x in returns}
    return returns


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


def resample_matched_timeseries(old_timeseries, new_sample_times, old_sample_times):
    good_samples = ~np.any(np.isnan(old_timeseries), axis=1)
    resampled_behavior = np.zeros((new_sample_times.shape[0], old_timeseries.shape[1]))
    for c in range(resampled_behavior.shape[1]):
        resampled_behavior[:, c] = np.interp(new_sample_times, old_sample_times[good_samples], old_timeseries[good_samples, c])
    return resampled_behavior


def center_from_first_n(A, n=100):
    return A[n:] - A[:n].mean(axis=0)


def align_column_spaces(A, B):
    # R = argmin(lambda omega: norm(omega @ A - B))
    A, B = A.T, B.T
    C = A @ B.T
    u, s, vh = np.linalg.svd(C)
    R = vh.T @ u.T
    return (R @ A).T, (B).T


def column_space_distance(Q1, Q2):
    Q1_rotated, Q2 = align_column_spaces(Q1, Q2)
    return np.linalg.norm(Q1_rotated - Q2)
