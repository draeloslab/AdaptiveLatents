import numpy as np
import os
import pickle
import json
from proSVD import proSVD
import bubblewrap
from tqdm import tqdm
import hashlib
import h5py
from pynwb import NWBHDF5IO
from skimage.transform import resize
from scipy.io import loadmat
from bubblewrap.config import CONFIG
import time



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


def save_to_cache(file, location=bubblewrap.config.CONFIG["data_path"] / "cache"):
    if not os.path.exists(location):
        os.mkdir(location)
    cache_index_file = os.path.join(location, f"{file}_index.pickle")
    try:
        with open(cache_index_file, 'rb') as fhan:
            cache_index = pickle.load(fhan)
    except FileNotFoundError:
        cache_index = {}

    def decorator(original_function):
        if not bubblewrap.config.CONFIG["attempt_to_cache"]:
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
    dataset = np.load(os.path.join(bubblewrap.config.CONFIG["data_path"], filename))
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
    bw = bubblewrap.Bubblewrap(dim=input_arr.shape[1], **bw_params)
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
    bw = bubblewrap.Bubblewrap(dim=input_arr.shape[1], **bw_params)
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


def construct_indy_data(dataset, bin_width=.03):
    """
    bin_width is in seconds
    """

    fhan = h5py.File(CONFIG["data_path"] / dataset, 'r')

    # this is a first pass I'm using to find the first and last spikes
    l = []
    for j in range(fhan['spikes'].shape[1]):
        for i in range(fhan['spikes'].shape[0]):
            v = np.squeeze(fhan[fhan['spikes'][i, j]])
            if v[0] > 50:  # empy channels have one spike very early; we want the other channels
                l.append(v)

    # this finds the first and last spikes in the dataset, so we can set our bin boundaries
    ll = [leaf for tree in l for leaf in tree]  # puts all the spike times into a flat list
    stop = np.ceil(max(ll))
    start = np.floor(min(ll))

    # this creates the bins we'll use to group spikes
    bins = np.arange(start, stop, bin_width)
    bin_centers = np.convolve([.5, .5], bins, "valid")

    # columns of A are channels, rows are time bins
    A = np.zeros(shape=(bins.shape[0] - 1, len(l)))
    c = 0  # we need this because some channels are empty
    for j in range(fhan['spikes'].shape[1]):
        for i in range(fhan['spikes'].shape[0]):
            v = np.squeeze(fhan[fhan['spikes'][i, j]])
            if v[0] > 50:
                A[:, c], _ = np.histogram(np.squeeze(fhan[fhan['spikes'][i, j]]), bins=bins)
                c += 1

    # load behavior data
    raw_behavior = fhan['finger_pos'][:].T
    t = fhan["t"][0]

    # this resamples the behavior so it's in sync with the binned spikes
    behavior = np.zeros((bin_centers.shape[0], raw_behavior.shape[1]))
    for c in range(behavior.shape[1]):
        behavior[:, c] = np.interp(bin_centers, t, raw_behavior[:, c])

    mask = bin_centers > 70 # behavior is near-constant before 70 seconds
    bin_centers, behavior, A = bin_centers[mask], behavior[mask], A[mask]

    raw_behavior[:, 0] -= 8.5
    raw_behavior[:, 0] *= 10

    return A, raw_behavior, bin_centers, t


@save_to_cache("buzaki_data")
def construct_buzaki_data(base, bin_size):
    parent_folder = CONFIG["data_path"] / 'buzaki'
    def read_int_file(fname):
        with open(fname) as fhan:
            ret = []
            for line in fhan:
                line = int(line.strip())
                ret.append(line)
            return ret

    shanks = []
    for n in range(30):
        shanks.append(os.path.isfile(parent_folder / base / f"{base}.clu.{n}"))

    assert not any(shanks[20:])
    shanks = np.nonzero(shanks)[0]

    sampling_rate = 20_000
    clusters_to_ignore = {0,1}

    shank_datas = []
    cluster_mapping = {} # this will be a bijective dictionary between the (shank, cluster) and unit_number (also nan entries)

    min_time = float("inf")
    max_time = 0
    used_columns = 0
    for shank in shanks:
        clusters = read_int_file(parent_folder / base / f"{base}.clu.{shank}")
        n_clusters = clusters[0]
        clusters = clusters[1:]

        # TODO: check if I should exclude the hash unit
        for cluster in np.unique(clusters):
            if cluster not in clusters_to_ignore:
                cluster_mapping[(shank, cluster)] = used_columns
                used_columns += 1
                cluster_mapping[cluster_mapping[(shank, cluster)]] = (shank, cluster)
            else:
                cluster_mapping[(shank, cluster)] = np.nan


        clusters = [cluster_mapping[(shank,c)] for c in clusters]
        times = read_int_file(parent_folder / base / f"{base}.res.{shank}")

        pairs = np.array([times, clusters]).T
        pairs = pairs[~np.isnan(pairs[:,1]),:]

        if len(pairs):
            pairs[:,0] /= sampling_rate


            min_time = min(min_time, pairs[:,0].min())
            max_time = max(max_time, pairs[:,0].max())


            shank_datas.append(pairs)

    bins = np.arange(min_time, max_time + bin_size, bin_size)
    bin_centers = np.convolve([.5, .5], bins, "valid")
    A = np.zeros((len(bins)-1, used_columns ))

    for shank_data in shank_datas:
        max_lower_bound = 0
        last_time = 0
        for time, cluster in shank_data:
            assert time >= last_time
            while time > bins[max_lower_bound + 1]:
                max_lower_bound += 1
            A[max_lower_bound, int(cluster)] += 1
            last_time = time

    with open(parent_folder / base / f"{base}.whl", "r") as fhan:
        coords = [[] for _ in range(4)]
        for line in fhan:
            line = [float(x) for x in line[:-1].split("\t")]
            for i in range(4):
                coords[i].append(line[i])

    raw_behavior = np.array(coords).T

    sampling_rate = 39.06
    t = np.arange(raw_behavior.shape[0])/sampling_rate

    raw_behavior[raw_behavior == -1] = np.nan

    return A, raw_behavior, bin_centers, t

def construct_fly_data(dataset):
    with NWBHDF5IO(CONFIG["data_path"] / 'fly' / dataset, mode="r", load_namespaces=True) as fhan:
        file = fhan.read()
        A = file.processing["ophys"].data_interfaces["DfOverF"].roi_response_series['RoiResponseSeries'].data[:]
        beh = file.processing['behavioral state'].data_interfaces['behavioral state'].data[:]
        t = file.processing['behavioral state'].data_interfaces['behavioral state'].timestamps[:]
        # t = file.processing["behavior"].data_interfaces["ball_motion"].timestamps[:]
    return A, beh, t, t

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

datasets = {
'fly': ['2019_06_28_fly2.nwb',
 '2019_07_01_fly2.nwb',
 '2019_08_07_fly2.nwb',
 '2019_08_14_fly1.nwb',
 '2019_08_14_fly2.nwb',
 '2019_08_14_fly3_2.nwb',
 '2019_08_20_fly2.nwb',
 '2019_08_20_fly3.nwb',
 '2019_10_02_fly2.nwb',
 '2019_10_10_fly3.nwb',
 '2019_10_14_fly2.nwb',
 '2019_10_14_fly3.nwb',
 '2019_10_14_fly4.nwb',
 '2019_10_18_fly2.nwb',
 '2019_10_18_fly3.nwb',
 '2019_10_21_fly1.nwb'
 ],
"buzaki": ["Mouse12-120806", "Mouse12-120807", "Mouse24-131216"],
"indy": ['indy_20160407_02.mat']
}
def main():
    obs, beh = get_from_saved_npz("jpca_reduced_sc.npz")
    obs = zscore(prosvd_data(obs, output_d=2, init_size=30), init_size=3)
    alphas = bwrap_alphas(input_arr=obs, bw_params=bubblewrap.default_parameters.default_jpca_dataset_parameters)
    print(alphas)


if __name__ == '__main__':
    main()
