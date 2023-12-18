import numpy as np
import os
import pickle
import json
from proSVD import proSVD
import adaptive_latents
from tqdm import tqdm
import hashlib
from skimage.transform import resize
from scipy.io import loadmat
from adaptive_latents.config import CONFIG

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.shape[0] > 1000:
                n_samples = 200
                row_samples = [round(x * (obj.shape[0]-1)) for x in np.linspace(0,1, n_samples)]
                obj = obj[row_samples]
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def make_hashable(x):
    return json.dumps(x, sort_keys=True, cls=NumpyEncoder).encode()

def make_hashable_and_hash(x):
    return int(hashlib.sha1(make_hashable(x)).hexdigest(), 16)


def save_to_cache(file, location=adaptive_latents.config.CONFIG["data_path"] / "cache"):
    if not os.path.exists(location):
        os.mkdir(location)
    cache_index_file = os.path.join(location, f"{file}_index.pickle")
    try:
        with open(cache_index_file, 'rb') as fhan:
            cache_index = pickle.load(fhan)
    except FileNotFoundError:
        cache_index = {}

    def decorator(original_function):
        if not adaptive_latents.config.CONFIG["attempt_to_cache"]:
            return original_function

        def new_function(**kwargs):
            kwargs_as_key = make_hashable_and_hash(kwargs)

            if kwargs_as_key not in cache_index or not os.path.exists(location / cache_index[kwargs_as_key]):
                result = original_function(**kwargs)

                hstring = str(kwargs_as_key)[-15:]
                cache_file = os.path.join(location,f"{file}_{hstring}.pickle")
                if CONFIG["show_cache_files"]:
                    print(f"caching value in: {cache_file}")
                with open(cache_file, "wb") as fhan:
                    pickle.dump(result, fhan)

                cache_index[kwargs_as_key] = cache_file
                with open(cache_index_file, 'bw') as fhan:
                    pickle.dump(cache_index, fhan)
            elif CONFIG["validate_cache"]: # TODO: this doesn't work
                result = original_function(**kwargs)
                with open(os.path.join(location, cache_index[kwargs_as_key]), 'rb') as fhan:
                    assert make_hashable_and_hash(result) == make_hashable_and_hash(pickle.load(fhan))

            with open(os.path.join(location, cache_index[kwargs_as_key]), 'rb') as fhan:
                if CONFIG["show_cache_files"]:
                    print(f"retreiving cache from: {cache_index[kwargs_as_key]}")
                return pickle.load(fhan)

        return new_function
    return decorator


def get_from_saved_npz(filename):
    dataset = np.load(os.path.join(adaptive_latents.config.CONFIG["data_path"], filename))
    beh = dataset['x']

    if len(dataset['y'].shape) == 3:
        obs = dataset['y'][0]
    else:
        obs = dataset['y']

    return obs, beh.reshape([obs.shape[0], -1])

@save_to_cache("prosvd_data")
def prosvd_data(input_arr, output_d, init_size):
    pro = proSVD(k=output_d)
    pro.initialize(input_arr[:init_size].T)

    output = []
    for i in tqdm(range(init_size, len(input_arr))):
        obs = input_arr[i:i + 1, :]
        if np.any(np.isnan(obs)):
            output.append(np.zeros(output_d) * np.nan)
            continue
        pro.preupdate()
        pro.updateSVD(obs.T)
        pro.postupdate()

        obs = obs @ pro.Q

        output.append(obs)
    return np.array(output).reshape((-1, output_d))


def zscore(input_arr, init_size=6, clip=True):
    mean = 0
    m2 = 1e-4
    output = []
    count = 0
    for i, x in enumerate(tqdm(input_arr)):
        if np.any(np.isnan(x)):
            output.append(x * np.nan)
            continue

        if count >= init_size:
            std = np.sqrt(m2 / (count - 1))
            output.append((x - mean) / std)

        delta = x - mean
        mean += delta / (count + 1)
        m2 += delta * (x - mean)
        count += 1
    output = np.array(output)

    if clip:
        output[output > 15] = 15
        output[output < -15] = -15
    return output

# todo: some rank-version of zscore?


def shuffle_time(*input_arr_list, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    p = rng.permutation(input_arr_list[0].shape[0])

    return [x[p,:] for x in input_arr_list]


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
    returns = {x:[] for x in nsteps}
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
        maxlen: a maximum length to trim them all down to (defaults to the shortest of the lengths of the iterables)

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



def resample_behavior(raw_behavior,bin_centers,t):
    good_samples = ~np.any(np.isnan(raw_behavior), axis=1)
    resampled_behavior = np.zeros((bin_centers.shape[0], raw_behavior.shape[1]))
    for c in range(resampled_behavior.shape[1]):
        resampled_behavior[:,c] = np.interp(bin_centers, t[good_samples], raw_behavior[good_samples,c])
    return resampled_behavior

@save_to_cache("generate_musal_dataset")
def generate_musal_dataset(cam=1, video_target_dim=100, resize_factor=1, prosvd_init_size=100):
    """cam ∈ {1,2}"""
    ca_sampling_rate = 31
    video_sampling_rate = 30

    #### load A
    data_dir = CONFIG["data_path"] / "musal/their_data/2pData/Animals/mSM49/SpatialDisc/30-Jul-2018/"
    variables = loadmat(data_dir/"data.mat",  squeeze_me=True, simplify_cells=True)
    A = variables["data"]['dFOF']
    _,n_samples_per_trial,_ = A.shape
    A = np.vstack(A.T)

    #### load trial start and end times, in video frames
    def read_floats(file):
        with open(file) as fhan:
            text = fhan.read()
            return [float(x) for x in text.split(",")]
    on_times = read_floats(CONFIG["data_path"] / "musal" / "trialOn.txt")
    off_times = read_floats(CONFIG["data_path"] / "musal" / "trialOff.txt")
    trial_edges = np.array([on_times, off_times]).T
    trial_edges = trial_edges[np.all(np.isfinite(trial_edges), axis=1)].astype(int)

    #### load video
    root_dir = data_dir / "BehaviorVideo"

    start_V = 0  # 29801
    end_V = trial_edges.max() # 89928
    used_V = end_V - start_V

    Wid, Hei = 320, 240
    Wid0, Hei0 = Wid//4, Hei//4

    # resized by half
    Data = np.zeros((used_V, Wid//resize_factor, Hei//resize_factor))

    for k in tqdm(range(16)):
        name = f'{root_dir}/SVD_Cam{cam}-Seg{k+1}.mat'
        # Load MATLAB .mat file
        mat_contents = loadmat(name)
        V = mat_contents['V'] # (89928, 500)
        U = mat_contents['U'] # (500, 4800)

        VU = V[start_V:end_V, :].dot(U) # (T, 4800)
        seg = VU.reshape((used_V, Wid0, Hei0))
        Wid1, Hei1 = Wid0//resize_factor, Hei0//resize_factor
        seg = resize(seg, (used_V, Wid1, Hei1), mode='constant')

        i, j = k//4, (k%4)
        Data[:, i * Wid1: (i+ 1) * Wid1, j*Hei1 : (j + 1) * Hei1] = seg

    #### dimension reduce video
    t = np.arange(Data.shape[0])/video_sampling_rate
    d = np.array(Data.reshape(Data.shape[0],-1))
    del Data
    d = prosvd_data(input_arr=d, output_d=video_target_dim, init_size=prosvd_init_size)
    t, d = clip(t, d)

    #### define times
    ca_times = np.hstack([np.linspace(*trial_edges[i], n_samples_per_trial) for i in range(len(trial_edges))])
    ca_times = ca_times/video_sampling_rate

    return A, d, ca_times, t

def main():
    obs, beh = get_from_saved_npz("jpca_reduced_sc.npz")
    obs = zscore(prosvd_data(obs, output_d=2, init_size=30), init_size=3)
    alphas = bwrap_alphas(input_arr=obs, bw_params=adaptive_latents.default_parameters.default_jpca_dataset_parameters)
    print(alphas)


if __name__ == '__main__':
    main()
