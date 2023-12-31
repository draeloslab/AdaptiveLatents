import numpy as np
import h5py
import os
from scipy.io import loadmat
from tqdm import tqdm
from .functional import save_to_cache, clip, prosvd_data
from skimage.transform import resize
from pynwb import NWBHDF5IO
from adaptive_latents.config import CONFIG


individual_identifiers = {
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



def construct_indy_data(individual_identifier, bin_width):
    """
    bin_width is in seconds; a sensible default is 0.03
    """

    fhan = h5py.File(CONFIG["data_path"] / individual_identifier, 'r')

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
def construct_buzaki_data(individual_identifier, bin_width):
    """"bin_width is in seconds; a sensible default is 0.03"""
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
        shanks.append(os.path.isfile(parent_folder / individual_identifier / f"{individual_identifier}.clu.{n}"))

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
        clusters = read_int_file(parent_folder / individual_identifier / f"{individual_identifier}.clu.{shank}")
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
        times = read_int_file(parent_folder / individual_identifier / f"{individual_identifier}.res.{shank}")

        pairs = np.array([times, clusters]).T
        pairs = pairs[~np.isnan(pairs[:,1]),:]

        if len(pairs):
            pairs[:,0] /= sampling_rate


            min_time = min(min_time, pairs[:,0].min())
            max_time = max(max_time, pairs[:,0].max())


            shank_datas.append(pairs)

    bins = np.arange(min_time, max_time + bin_width, bin_width)
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

    with open(parent_folder / individual_identifier / f"{individual_identifier}.whl", "r") as fhan:
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

def construct_fly_data(individual_identifier):
    with NWBHDF5IO(CONFIG["data_path"] / 'fly' / individual_identifier, mode="r", load_namespaces=True) as fhan:
        file = fhan.read()
        A = file.processing["ophys"].data_interfaces["DfOverF"].roi_response_series['RoiResponseSeries'].data[:]
        beh = file.processing['behavioral state'].data_interfaces['behavioral state'].data[:]
        t = file.processing['behavioral state'].data_interfaces['behavioral state'].timestamps[:]
        # t = file.processing["behavior"].data_interfaces["ball_motion"].timestamps[:]
    return A, beh, t, t


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
