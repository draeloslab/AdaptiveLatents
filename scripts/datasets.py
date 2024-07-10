import numpy as np
import h5py
import os
from scipy.io import loadmat
from tqdm import tqdm
from adaptive_latents.transforms.utils import save_to_cache, clip, prosvd_data
from skimage.transform import resize
from pynwb import NWBHDF5IO
import pathlib

import fsspec
from fsspec.implementations.cached import CachingFileSystem
from dandi.dandiapi import DandiAPIClient

import datahugger

import bs4
import urllib.request

DATA_BASE_PATH = pathlib.Path(__file__).parent.resolve() / 'datasets'

individual_identifiers = {
    "indy": ['indy_20160407_02.mat'],
    "peyrache15": ["Mouse12-120806", "Mouse12-120807", "Mouse24-131216"],
    'schaffer23': ['2019_06_28_fly2.nwb',
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
    'jenkins': [],
    'musal':[],
    'nason20': [],
}

def construct_low21_data():
    dataset_base_path = DATA_BASE_PATH / 'low21'
    if len(list(dataset_base_path.glob("*.npy"))) == 0:
        datahugger.get("https://doi.org/10.17632/hntn6m2pgk.1", dataset_base_path)
    # TODO


def _download_odoherty21(dataset_base_path, filename="indy_20160407_02.mat"):
    try:
        file_url = f"https://zenodo.org/records/3854034/files/{filename}?download=1"""
        dataset_base_path.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url=file_url, filename=dataset_base_path / filename)
    except:
        datahugger.get("https://doi.org/10.5281/zenodo.3854034", dataset_base_path) # this is usually very slow


def construct_odoherty21_data(individual_identifier='indy_20160407_02.mat', bin_width=0.03):
    """
    bin_width is in seconds; a sensible default is 0.03
    """

    dataset_base_path = DATA_BASE_PATH / "odoherty21"
    if not (dataset_base_path / individual_identifier).is_file():
        _download_odoherty21(dataset_base_path, individual_identifier)

    fhan = h5py.File(dataset_base_path / individual_identifier, 'r')

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



def construct_schaffer23_data(individual_identifier):
    # https://doi.org/10.6084/m9.figshare.23749074

    dataset_base_path = DATA_BASE_PATH / 'schaffer23'

    if len(list(dataset_base_path.glob("*.nwb"))) == 0:
        datahugger.get("https://doi.org/10.6084/m9.figshare.23749074", dataset_base_path)

    with NWBHDF5IO(dataset_base_path / individual_identifier, mode="r", load_namespaces=True) as fhan:
        file = fhan.read()
        A = file.processing["ophys"].data_interfaces["DfOverF"].roi_response_series['RoiResponseSeries'].data[:]
        beh = file.processing['behavioral state'].data_interfaces['behavioral state'].data[:]
        t = file.processing['behavioral state'].data_interfaces['behavioral state'].timestamps[:]
        # t = file.processing["behavior"].data_interfaces["ball_motion"].timestamps[:]
    return A, beh, t, t

def construct_churchland22_data(bin_width):
    # https://pynwb.readthedocs.io/en/latest/tutorials/advanced_io/streaming.html
    dandiset_id = '000128'
    filepath = 'sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb'
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id, version_id='0.220113.0400').get_asset_by_path(filepath)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

    fs = fsspec.filesystem("http")

    # create a cache to save downloaded data to disk (optional)
    fs = CachingFileSystem(
        fs=fs,
        cache_storage=[DATA_BASE_PATH/"nwb_cache"],
    )

    with fs.open(s3_url, "rb") as f:
        with h5py.File(f) as file:
            with NWBHDF5IO(file=file) as fhan:
                nwb_in = fhan.read()
                units = nwb_in.units.to_dataframe()
                hand_pos = np.array(nwb_in.processing['behavior'].data_interfaces['hand_pos'].data)
                hand_t = np.array(nwb_in.processing['behavior'].data_interfaces['hand_pos'].timestamps)

    bin_edges = np.arange(units.iloc[0, 2][0, 0], units.iloc[0, 2][-1, -1] + bin_width, bin_width)

    A = np.zeros((len(bin_edges) - 1, units.shape[0]))

    for i in range(units.shape[0]):
        A[:, i], _ = np.histogram(units.iloc[i, 1], bin_edges)

    recorded_intervals = units.iloc[0, 2]

    interval_to_start_from = 0

    def intersection(start1, stop1, start2, stop2):
        return max(min(stop1, stop2) - max(start1, start2), 0)

    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_stop = bin_edges[i + 1]
        covered = 0
        for j in range(interval_to_start_from, recorded_intervals.shape[0]):
            interval_start, interval_stop = recorded_intervals[j]
            if bin_start > interval_stop:
                interval_to_start_from += 1
                continue
            if interval_start > bin_stop:
                break
            covered += intersection(bin_start, bin_stop, interval_start, interval_stop)

        if covered / bin_width < .9:
            A[i, :] = np.nan

    bin_centers = np.convolve(bin_edges, [.5, .5], 'valid')

    return A, hand_pos, bin_centers, hand_t


def construct_nason20_data(bin_width=0.15):
    "https://doi.org/10.7302/wwya-5q86"
    bin_width_in_ms = int(bin_width*1000)
    file = DATA_BASE_PATH / 'nason20'/ 'OnlineTrainingData.mat'
    mat = loadmat(file, squeeze_me=True, simplify_cells=True)
    data = mat['OnlineTrainingData']
    n_channels = data[0]['SpikingBandPower'].shape[1]

    for i in range(len(data) - 1):
        assert data[i + 1]['ExperimentTime'][0] - data[i]['ExperimentTime'][-1] == 3

    A = []
    t = []
    beh = []
    for i, trial in enumerate(data):
        A_spacer = np.nan * np.zeros((3,n_channels))
        t_spacer = np.arange(1,4) + trial['ExperimentTime'][-1]
        beh_spacer = t_spacer * np.nan
        if i == len(data)-1:
            A_spacer = np.zeros((0,n_channels))
            t_spacer = []
            beh_spacer = []
        sub_A_spaced = np.vstack([trial['SpikingBandPower'], A_spacer])
        sub_t_spaced = np.hstack([trial['ExperimentTime'], t_spacer])
        sub_beh_spaced = np.hstack([trial['FingerAngle'], beh_spacer])
        A.append(sub_A_spaced)
        t.append(sub_t_spaced)
        beh.append(sub_beh_spaced)
    A = np.vstack(A)
    t = np.hstack(t) / 1000 # converts to seconds
    beh = np.hstack(beh)

    s = t > 1.260 # there's an early dead zone
    A, beh, t = A[s], beh[s], t[s]


    aug = np.column_stack([t,beh, A])
    binned_aug = aug[aug.shape[0] % bin_width_in_ms:,:].reshape(( -1, bin_width_in_ms, aug.shape[1]))
    t = binned_aug[:,:,0].max(axis=1)
    beh = np.nanmean(binned_aug[:,:,1], axis=1)
    A = np.nanmean(binned_aug[:,:,2:], axis=1)

    return A, beh, t, t

@save_to_cache("peyrache15_data")
def construct_peyrache15_data(individual_identifier, bin_width):
    """"bin_width is in seconds; a sensible default is 0.03"""
    # http://dx.doi.org/10.6080/K0G15XS1
    parent_folder = DATA_BASE_PATH / 'peyrache15'
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
def construct_unpublished24_data(include_position=True, include_velocity=False, include_acceleration=False):
    mat = loadmat(DATA_BASE_PATH / 'Chestek' / 'jgould_first_extraction.mat', squeeze_me=True, simplify_cells=True)
    pre_smooth_beh = mat["feats"][1]
    pre_smooth_A = mat["feats"][0]
    pre_smooth_t = mat["feats"][2] / 1000

    pre_smooth_beh = pre_smooth_beh.reshape((pre_smooth_beh.shape[0], 3, 5))

    nonzero_columns = pre_smooth_beh.std(axis=0) > 0
    assert np.all(~(nonzero_columns[0, :] ^ nonzero_columns))  # checks that fingers always have the same values
    pre_smooth_beh = pre_smooth_beh[:, :,
                     nonzero_columns[0, :]]  # the booleans select for position, velocity, and acceleration
    pre_smooth_beh = pre_smooth_beh[:, [include_position, include_velocity, include_acceleration], :].reshape(pre_smooth_beh.shape[0],
                                                                        -1)  # the three booleans select for position, velocity, and acceleration
    kernel = np.exp(np.linspace(0, -1, 5))
    kernel /= kernel.sum()


    mode = 'valid'
    A = np.column_stack([np.convolve(kernel, column, mode) for column in pre_smooth_A.T])
    t = np.convolve(np.hstack([[1],kernel[:-1]*0]), pre_smooth_t, mode)
    beh = pre_smooth_beh


    # pre_prosvd_A = center_from_first_n(pre_center_A, 100)
    # pre_prosvd_A, pre_prosvd_beh, pre_prosvd_t = clip(pre_prosvd_A, pre_prosvd_beh, pre_prosvd_t)
    #
    # pre_jpca_A = prosvd_data(input_arr=pre_prosvd_A, output_d=4, init_size=50)
    # pre_jpca_A, pre_jpca_t, pre_jpca_beh = clip(pre_jpca_A, pre_prosvd_t, pre_prosvd_beh)
    #
    # A, beh, t = pre_jpca_A, pre_jpca_beh, pre_jpca_t
    return A, beh, t, t

@save_to_cache("generate_musall19_dataset")
def generate_musall19_dataset(cam=1, video_target_dim=100, resize_factor=1, prosvd_init_size=100):
    """
    https://doi.org/10.1038/s41593-019-0502-4
    cam ∈ {1,2}
    """

    ca_sampling_rate = 31
    video_sampling_rate = 30

    #### load A
    data_dir = DATA_BASE_PATH / "musal/their_data/2pData/Animals/mSM49/SpatialDisc/30-Jul-2018/"
    variables = loadmat(data_dir/"data.mat",  squeeze_me=True, simplify_cells=True)
    A = variables["data"]['dFOF']
    _,n_samples_per_trial,_ = A.shape
    A = np.vstack(A.T)

    #### load trial start and end times, in video frames
    def read_floats(file):
        with open(file) as fhan:
            text = fhan.read()
            return [float(x) for x in text.split(",")]
    on_times = read_floats(DATA_BASE_PATH / "musal" / "trialOn.txt")
    off_times = read_floats(DATA_BASE_PATH / "musal" / "trialOff.txt")
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
    d = prosvd_data(input_arr=d, output_d=video_target_dim, init_size=prosvd_init_size, centering=False)
    t, d = clip(t, d)

    #### define times
    ca_times = np.hstack([np.linspace(*trial_edges[i], n_samples_per_trial) for i in range(len(trial_edges))])
    ca_times = ca_times/video_sampling_rate

    return A, d, ca_times, t

if __name__ == '__main__':
    construct_schaffer23_data(individual_identifier=individual_identifiers['schaffer23'][0])

"""
low21
indy
fly
jenkins

buzsaki
musal
unpublished24
"""