import numpy as np
import h5py
import os
from scipy.io import loadmat
from tqdm import tqdm
from adaptive_latents.utils import save_to_cache, clip
from adaptive_latents import NumpyTimedDataSource, proSVD, CONFIG
from skimage.transform import resize
from pynwb import NWBHDF5IO
import pathlib
import pandas as pd
from abc import ABC, abstractmethod
import sys
import scipy.io
import matplotlib
import pims
from PIL import Image

import enum

from contextlib import contextmanager

import fsspec
from fsspec.implementations.cached import CachingFileSystem
from dandi.dandiapi import DandiAPIClient

import datahugger

import urllib.request

DATA_BASE_PATH = CONFIG['dataset_path']


class ModelOrganism(enum.Enum):
    FLY = 'Drosophila melanogaster'
    MONKEY = 'Macaca mulatta'
    RAT = 'Rattus rattus'
    MOUSE = 'Mus musculus'
    FINCH = 'Taeniopygia castanotis'
    FISH = 'Danio rerio'


class Dataset(ABC):
    neural_data: NumpyTimedDataSource
    behavioral_data: NumpyTimedDataSource
    opto_stimulations = None

    @property
    @abstractmethod
    def doi(self):
        pass

    @property
    @abstractmethod
    def automatically_downloadable(self):
        pass

    @property
    @abstractmethod
    def model_organism(self) -> ModelOrganism:
        pass

    @abstractmethod
    def acquire(self, *args, **kwargs):
        pass


class DandiDataset(Dataset):
    automatically_downloadable = True

    @property
    @abstractmethod
    def dandiset_id(self):
        pass

    @property
    @abstractmethod
    def version_id(self):
        pass

    @contextmanager
    def acquire(self, asset_path):
        # https://pynwb.readthedocs.io/en/latest/tutorials/advanced_io/streaming.html
        with DandiAPIClient() as client:
            asset = client.get_dandiset(self.dandiset_id, version_id=self.version_id).get_asset_by_path(asset_path)
            s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

        fs = fsspec.filesystem("http")
        fs = CachingFileSystem(
            fs=fs,
            cache_storage=[DATA_BASE_PATH / "nwb_cache"],
        )

        with fs.open(s3_url, "rb") as f:
            with h5py.File(f) as file:
                fhan = NWBHDF5IO(file=file)
                yield fhan


class Odoherty21Dataset(DandiDataset):
    doi = 'https://dandiarchive.org/dandiset/000129/draft/'
    model_organism = ModelOrganism.MONKEY
    dandiset_id = "000129"
    version_id = None

    dataset_base_path = DATA_BASE_PATH / "odoherty21"
    automatically_downloadable = True

    def __init__(self, bin_width=0.03, downsample_behavior=True, neural_lag=0, drop_third_coord=False):
        self.bin_width = bin_width
        self.downsample_behavior = downsample_behavior
        self.drop_third_coord = drop_third_coord
        self.neural_lag = neural_lag
        assert self.neural_lag >= 0

        self.units, self.finger_pos, self.finger_vel, self.finger_t, A, bin_ends = self.construct()
        self.neural_data = NumpyTimedDataSource(A, bin_ends)
        self.behavioral_data = NumpyTimedDataSource(self.finger_pos, self.finger_t)

        self.beh_pos = NumpyTimedDataSource(self.finger_pos, self.finger_t)
        self.beh_vel = NumpyTimedDataSource(self.finger_vel, self.finger_t)
        self.beh_pos_vel = NumpyTimedDataSource(np.hstack([self.finger_pos, self.finger_vel]), self.finger_t)

    def construct(self):
        with self.acquire("sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb") as fhan:
            ds = fhan.read()
            units = ds.units.to_dataframe()
            finger_pos = ds.processing['behavior'].data_interfaces['finger_pos'].data[:]
            finger_pos_t = np.arange(finger_pos.shape[0]) * ds.processing['behavior'].data_interfaces['finger_pos'].conversion
            finger_vel = ds.processing['behavior'].data_interfaces['finger_vel'].data[:]
            finger_vel_t = np.arange(finger_vel.shape[0]) * ds.processing['behavior'].data_interfaces['finger_vel'].conversion

        start_time = units.iloc[0, 2].min()
        end_time = units.iloc[0, 2].max()
        bins = np.arange(start_time, end_time, self.bin_width)
        bin_ends = bins[1:]

        A = np.zeros(shape=(bins.shape[0] - 1, len(units)))

        for i, (_, row) in enumerate(units.iterrows()):
            A[:, i], _ = np.histogram(row['spike_times'], bins=bins)


        factor = 4
        if self.downsample_behavior:
            finger_pos = finger_pos[::factor]
            finger_pos_t = finger_pos_t[::factor]
            finger_vel = finger_vel[::factor]
            finger_vel_t = finger_vel_t[::factor]

        bin_ends = bin_ends + self.neural_lag
        assert (finger_pos_t == finger_vel_t).all()
        finger_t = finger_pos_t

        if self.drop_third_coord:
            finger_pos = finger_pos[:,:2]
            finger_vel = finger_vel[:,:2]


        return units, finger_pos, finger_vel, finger_t, A, bin_ends


class Schaffer23Datset(Dataset):
    doi = 'https://doi.org/10.6084/m9.figshare.23749074'
    model_organism = ModelOrganism.FLY
    dataset_base_path = DATA_BASE_PATH / 'schaffer23'
    automatically_downloadable = True
    sub_datasets = (
        '2019_06_28_fly2.nwb', '2019_07_01_fly2.nwb', '2019_08_07_fly2.nwb', '2019_08_14_fly1.nwb',
        '2019_08_14_fly2.nwb', '2019_08_14_fly3_2.nwb', '2019_08_20_fly2.nwb', '2019_08_20_fly3.nwb',
        '2019_10_02_fly2.nwb', '2019_10_10_fly3.nwb', '2019_10_14_fly2.nwb', '2019_10_14_fly3.nwb',
        '2019_10_14_fly4.nwb', '2019_10_18_fly2.nwb', '2019_10_18_fly3.nwb', '2019_10_21_fly1.nwb'
    )

    def __init__(self, sub_dataset_identifier=sub_datasets[0]):
        self.sub_dataset = sub_dataset_identifier
        A, beh, t, t = self.construct(sub_dataset_identifier)
        self.neural_data = NumpyTimedDataSource(A,t)
        self.behavioral_data = NumpyTimedDataSource(beh,t)

    def construct(self, sub_dataset_identifier):
        with self.acquire(sub_dataset_identifier) as fhan:
            file = fhan.read()
            A = file.processing["ophys"].data_interfaces["DfOverF"].roi_response_series['RoiResponseSeries'].data[:]
            beh = file.processing['behavioral state'].data_interfaces['behavioral state'].data[:]
            t = file.processing['behavioral state'].data_interfaces['behavioral state'].timestamps[:]
            # t = file.processing["behavior"].data_interfaces["ball_motion"].timestamps[:]
        return A, beh, t, t

    def acquire(self, sub_dataset_identifier):
        if len(list(self.dataset_base_path.glob("*.nwb"))) == 0:
            datahugger.get(self.doi, self.dataset_base_path)

        if sub_dataset_identifier is not None:
            return NWBHDF5IO(self.dataset_base_path / sub_dataset_identifier, mode="r", load_namespaces=True)


class Churchland10Dataset(DandiDataset):
    doi = 'https://doi.org/10.48324/dandi.000128/0.220113.0400'
    model_organism = ModelOrganism.MONKEY
    dandiset_id = '000128'
    version_id = '0.220113.0400'
    automatically_downloadable = True

    def __init__(self, bin_width=0.03):
        self.bin_width = bin_width
        neural_data, hand_position, nerual_t, hand_t = self.construct()
        self.neural_data = NumpyTimedDataSource(neural_data, nerual_t)
        self.behavioral_data = NumpyTimedDataSource(hand_position, hand_t)

    def construct(self,):
        with self.acquire('sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb') as fhan:
            nwb_in = fhan.read()
            units = nwb_in.units.to_dataframe()
            hand_pos = np.array(nwb_in.processing['behavior'].data_interfaces['hand_pos'].data)
            hand_t = np.array(nwb_in.processing['behavior'].data_interfaces['hand_pos'].timestamps)

        bin_edges = np.arange(units.iloc[0, 2][0, 0], units.iloc[0, 2][-1, -1] + self.bin_width, self.bin_width)

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

            if covered / self.bin_width < .9:
                A[i, :] = np.nan

        bin_ends = bin_edges[1:]

        return A, hand_pos, bin_ends, hand_t


class TostadoMarcos24Dataset(DandiDataset):
    doi = 'https://dandiarchive.org/dandiset/001046/draft'
    dandiset_id = '001046'
    version_id = 'draft'
    automatically_downloadable = True
    model_organism = ModelOrganism.FINCH
    sub_datasets = ['27', '26', '28']

    def __init__(self, sub_dataset_identifier=sub_datasets[0], bin_size=0.03):
        self.sub_dataset = sub_dataset_identifier
        self.bin_size = bin_size
        self.tx, self.vocalizations, self.neural_data, self.behavioral_data = self.construct(self.sub_dataset)

    def construct(self, sub_dataset_identifier):
        with self.acquire(f"sub-Finch-z-r12r13-21-held-in-calib/sub-Finch-z-r12r13-21-held-in-calib_ses-202106{sub_dataset_identifier}.nwb") as fhan:
            nwb = fhan.read()
            # TODO: make it possible to pass around TimeSeries with the HDF5 dereferenced
            tx = nwb.acquisition['tx'].data[:]
            tx_t = nwb.acquisition['tx'].timestamps[:]
            vocalizations = NumpyTimedDataSource.from_nwb_timeseries(nwb.acquisition['vocalizations'])
            trials = nwb.intervals['trials'].to_dataframe()


        # make FR matrix for neural data
        dt_s = np.diff(tx_t)
        dt = np.median(dt_s)
        assert dt_s.std() / dt < .0001

        bins = np.linspace(tx_t[0], tx_t[-1], int((tx_t[-1] - tx_t[0]) // self.bin_size) + 1)
        A = np.empty(shape=(bins.size - 1, tx.shape[1]))
        for i in range(A.shape[1]):
            counts, _ = np.histogram(tx_t[tx[:, i].nonzero()[0]], bins=bins)
            A[:, i] = counts
        t = np.convolve([.5, .5], bins, 'valid')
        neural_data = NumpyTimedDataSource(A, t)

        # make spectrogram matrix
        times = []
        spectral_data = []
        for idx, row in trials.iterrows():
            times.extend(row['spectrogram_times'] + row['start_time'])
            spectral_data.extend(row['spectrogram_values'].T)

        spectral_data = np.array(spectral_data)
        times = np.array(times)
        behavioral_data = NumpyTimedDataSource(spectral_data, times)

        return tx, vocalizations, neural_data, behavioral_data

    def play_audio(self):
        import IPython.display as ipd
        x = self.vocalizations.a.flatten()
        t = self.vocalizations.t.flatten()
        return ipd.Audio(x, rate=round(1 / np.median(np.diff(t))))

    def plot_recalculated_spectrogram(self, ax):
        import scipy.signal as ss

        x = self.vocalizations.a.flatten()
        t = self.vocalizations.t.flatten()

        dt = np.median(np.diff(t))
        Fs = 1 / dt

        window_length_in_s = .01
        window_length_in_samples = int(window_length_in_s // dt)
        window = ss.windows.tukey(window_length_in_samples)
        SFT = ss.ShortTimeFFT(win=window, hop=window_length_in_samples, fs=Fs)

        Sx = SFT.stft(x)

        N = len(t)
        # fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
        t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
        ax.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
                       rf"$\Delta t = {SFT.delta_t:g}\,$s)",
                ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
                       rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
                xlim=(t_lo, t_hi))

        im1 = ax.imshow((abs(Sx)), origin='lower', aspect='auto',
                         extent=SFT.extent(N), cmap='viridis')

        for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
                         (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
            ax.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)

        for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line:
            ax.axvline(t_, color='y', linestyle='--', alpha=0.5)

        fig = ax.get_figure()
        fig.colorbar(im1, label="Magnitude $|S_x(t, f)|$")
        fig.tight_layout()




class Nason20Dataset(Dataset):
    doi = 'https://doi.org/10.7302/wwya-5q86'
    directory_name = 'nason20'
    model_organism = ModelOrganism.MONKEY
    dataset_base_path = DATA_BASE_PATH / directory_name
    automatically_downloadable = False

    def __init__(self, bin_width=0.15):
        self.bin_width = bin_width
        a, beh, t, t = self.construct()
        self.neural_data = NumpyTimedDataSource(a, t)
        self.behavioral_data = NumpyTimedDataSource(beh, t)

    def acquire(self):
        file = self.dataset_base_path / 'OnlineTrainingData.mat'
        if not file.is_file():
            print(f"""\
Please manually download the OnlineTrainingData.mat file from {self.doi}.
Then put it in '{self.dataset_base_path}'.
""")
            raise FileNotFoundError()
        return loadmat(file, squeeze_me=True, simplify_cells=True)

    def construct(self):
        bin_width_in_ms = int(self.bin_width * 1000)

        mat = self.acquire()
        data = mat['OnlineTrainingData']
        n_channels = data[0]['SpikingBandPower'].shape[1]

        for i in range(len(data) - 1):
            assert data[i + 1]['ExperimentTime'][0] - data[i]['ExperimentTime'][-1] == 3

        A = []
        t = []
        beh = []
        for i, trial in enumerate(data):
            A_spacer = np.nan * np.zeros((3, n_channels))
            t_spacer = np.arange(1, 4) + trial['ExperimentTime'][-1]
            beh_spacer = t_spacer * np.nan
            if i == len(data) - 1:
                A_spacer = np.zeros((0, n_channels))
                t_spacer = []
                beh_spacer = []
            sub_A_spaced = np.vstack([trial['SpikingBandPower'], A_spacer])
            sub_t_spaced = np.hstack([trial['ExperimentTime'], t_spacer])
            sub_beh_spaced = np.hstack([trial['FingerAngle'], beh_spacer])
            A.append(sub_A_spaced)
            t.append(sub_t_spaced)
            beh.append(sub_beh_spaced)
        A = np.vstack(A)
        t = np.hstack(t) / 1000  # converts to seconds
        beh = np.hstack(beh)

        s = t > 1.260  # there's an early dead zone
        A, beh, t = A[s], beh[s], t[s]

        aug = np.column_stack([t, beh, A])
        binned_aug = aug[aug.shape[0] % bin_width_in_ms:, :].reshape((-1, bin_width_in_ms, aug.shape[1]))
        t = binned_aug[:, :, 0].max(axis=1)
        beh = np.nanmean(binned_aug[:, :, 1], axis=1)
        A = np.nanmean(binned_aug[:, :, 2:], axis=1)

        return A, beh, t, t


class Peyrache15Dataset(Dataset):
    doi = 'http://dx.doi.org/10.6080/K0G15XS1'
    model_organism = ModelOrganism.MOUSE
    dataset_base_path = DATA_BASE_PATH / 'peyrache15'
    automatically_downloadable = False
    sub_datasets = ("Mouse12-120806", "Mouse12-120807", "Mouse24-131216")

    def __init__(self, sub_dataset_identifier=sub_datasets[0], bin_width=0.03):
        self.sub_dataset = sub_dataset_identifier
        self.bin_width = bin_width
        A, raw_behavior, a_t, beh_t = self.construct(sub_dataset_identifier)
        self.neural_data = NumpyTimedDataSource(A, a_t)
        self.behavioral_data = NumpyTimedDataSource(raw_behavior, beh_t)

    def acquire(self, sub_dataset_identifier):
        if not (self.dataset_base_path / sub_dataset_identifier).is_dir():
            print(f"""\
Please download {sub_dataset_identifier} from {self.doi} and put it in {self.dataset_base_path}.
""")
            raise FileNotFoundError()

    def construct(self, sub_dataset_identifier):
        self.acquire(sub_dataset_identifier)

        @save_to_cache("peyrache15_data")
        def static_construct(sub_dataset_identifier, bin_width):
            def read_int_file(fname):
                with open(fname) as fhan:
                    ret = []
                    for line in fhan:
                        line = int(line.strip())
                        ret.append(line)
                    return ret

            shanks = []
            for n in range(30):
                shanks.append(os.path.isfile(self.dataset_base_path / sub_dataset_identifier / f"{sub_dataset_identifier}.clu.{n}"))

            assert not any(shanks[20:])
            shanks = np.nonzero(shanks)[0]

            sampling_rate = 20_000
            clusters_to_ignore = {0, 1}

            shank_datas = []
            cluster_mapping = {}  # this will be a bijective dictionary between the (shank, cluster) and unit_number (also nan entries)

            min_time = float("inf")
            max_time = 0
            used_columns = 0
            for shank in shanks:
                clusters = read_int_file(self.dataset_base_path / sub_dataset_identifier / f"{sub_dataset_identifier}.clu.{shank}")
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

                clusters = [cluster_mapping[(shank, c)] for c in clusters]
                times = read_int_file(self.dataset_base_path / sub_dataset_identifier / f"{sub_dataset_identifier}.res.{shank}")

                pairs = np.array([times, clusters]).T
                pairs = pairs[~np.isnan(pairs[:, 1]), :]

                if len(pairs):
                    pairs[:, 0] /= sampling_rate

                    min_time = min(min_time, pairs[:, 0].min())
                    max_time = max(max_time, pairs[:, 0].max())

                    shank_datas.append(pairs)

            bins = np.arange(min_time, max_time + bin_width, bin_width)
            bin_ends = bins[1:]
            A = np.zeros((len(bins) - 1, used_columns))

            for shank_data in shank_datas:
                max_lower_bound = 0
                last_time = 0
                for time, cluster in shank_data:
                    assert time >= last_time
                    while time > bins[max_lower_bound + 1]:
                        max_lower_bound += 1
                    A[max_lower_bound, int(cluster)] += 1
                    last_time = time

            with open(self.dataset_base_path / sub_dataset_identifier / f"{sub_dataset_identifier}.whl", "r") as fhan:
                coords = [[] for _ in range(4)]
                for line in fhan:
                    line = [float(x) for x in line[:-1].split("\t")]
                    for i in range(4):
                        coords[i].append(line[i])

            raw_behavior = np.array(coords).T

            sampling_rate = 39.06
            t = np.arange(raw_behavior.shape[0]) / sampling_rate

            raw_behavior[raw_behavior == -1] = np.nan

            return A, raw_behavior, bin_ends, t

        return static_construct(sub_dataset_identifier, self.bin_width)


class Temmar24uDataset(Dataset):
    doi = None
    model_organism = ModelOrganism.MONKEY
    dataset_base_path = DATA_BASE_PATH / 'temmar24u'
    automatically_downloadable = False

    def __init__(self, include_position=True, include_velocity=False, include_acceleration=False):
        self.include_position = include_position
        self.include_velocity = include_velocity
        self.include_acceleration = include_acceleration
        self.bin_width = .05  # in seconds, this is in jgould_first_extraction.mat

        neural_data, behavioral_data, neural_t, behavioral_t = self.construct()
        self.neural_data = NumpyTimedDataSource(neural_data, neural_t)
        self.behavioral_data = NumpyTimedDataSource(behavioral_data, behavioral_t)

    def acquire(self):
        file = self.dataset_base_path / 'jgould_first_extraction.mat'
        if not file.is_file():
            # todo: possibly run jgould_first_extraction here?
            print("""\
Talk to the Chestek lab to get access to this data, and Jonathan Gould for the specifics of how this file was generated.
This data will eventually be published, after which there should be an easier way to download it.
""")
            raise FileNotFoundError()
        return loadmat(file, squeeze_me=True, simplify_cells=True)

    def construct(self):
        mat = self.acquire()

        pre_smooth_beh = mat["feats"][1]
        pre_smooth_A = mat["feats"][0]
        pre_smooth_t = mat["feats"][2] / 1000

        pre_smooth_beh = pre_smooth_beh.reshape((pre_smooth_beh.shape[0], 3, 5))

        nonzero_columns = pre_smooth_beh.std(axis=0) > 0
        assert np.all(~(nonzero_columns[0, :] ^ nonzero_columns))  # checks that fingers always have the same values
        pre_smooth_beh = pre_smooth_beh[:, :, nonzero_columns[0, :]]  # the booleans select for position, velocity, and acceleration
        pre_smooth_beh = pre_smooth_beh[:, [self.include_position, self.include_velocity, self.include_acceleration], :].reshape(pre_smooth_beh.shape[0], -1)  # the three booleans select for position, velocity, and acceleration
        kernel = np.exp(np.linspace(0, -1, 5))
        kernel /= kernel.sum()

        mode = 'valid'
        A = np.column_stack([np.convolve(kernel, column, mode) for column in pre_smooth_A.T])
        t = np.convolve(np.hstack([[1], kernel[:-1] * 0]), pre_smooth_t, mode)
        beh = pre_smooth_beh

        # pre_prosvd_A = center_from_first_n(pre_center_A, 100)
        # pre_prosvd_A, pre_prosvd_beh, pre_prosvd_t = clip(pre_prosvd_A, pre_prosvd_beh, pre_prosvd_t)
        #
        # pre_jpca_A = prosvd_data(input_arr=pre_prosvd_A, output_d=4, init_size=50)
        # pre_jpca_A, pre_jpca_t, pre_jpca_beh = clip(pre_jpca_A, pre_prosvd_t, pre_prosvd_beh)
        #
        # A, beh, t = pre_jpca_A, pre_jpca_beh, pre_jpca_t
        return A, beh, t, pre_smooth_t


class Musall19Dataset(Dataset):
    doi = 'https://doi.org/10.1038/s41593-019-0502-4'
    model_organism = ModelOrganism.MOUSE
    dataset_base_path = DATA_BASE_PATH / 'musall19'
    inner_data_path = dataset_base_path / "their_data/2pData/Animals/mSM49/SpatialDisc/30-Jul-2018"
    automatically_downloadable = False

    def __init__(self, cam=1, video_target_dim=100, resize_factor=1):
        self.cam = cam  # either 1 or 2
        self.video_target_dim = video_target_dim
        self.resize_factor = resize_factor

        A, d, ca_times, t = self.construct()
        self.neural_data = NumpyTimedDataSource(A, ca_times)
        self.behavioral_data = NumpyTimedDataSource(d, t)

    def construct(self):
        self.acquire()

        @save_to_cache("musall19_data")
        def static_construct(cam, video_target_dim, resize_factor):
            ca_sampling_rate = 31
            video_sampling_rate = 30

            #### load A
            variables = loadmat(self.inner_data_path / 'data.mat', squeeze_me=True, simplify_cells=True)
            A = variables["data"]['dFOF']
            _, n_samples_per_trial, _ = A.shape
            A = np.vstack(A.T)

            #### load trial start and end times, in video frames
            def read_floats(file):
                with open(file) as fhan:
                    text = fhan.read()
                    return [float(x) for x in text.split(",")]

            on_times = read_floats(self.dataset_base_path / "trialOn.txt")
            off_times = read_floats(self.dataset_base_path / "trialOff.txt")
            trial_edges = np.array([on_times, off_times]).T
            trial_edges = trial_edges[np.all(np.isfinite(trial_edges), axis=1)].astype(int)

            #### load video
            root_dir = self.inner_data_path / "BehaviorVideo"

            start_V = 0  # 29801
            end_V = trial_edges.max()  # 89928
            used_V = end_V - start_V

            Wid, Hei = 320, 240
            Wid0, Hei0 = Wid // 4, Hei // 4

            # resized by half
            Data = np.zeros((used_V, Wid // resize_factor, Hei // resize_factor))

            for k in tqdm(range(16)):
                name = f'{root_dir}/SVD_Cam{cam}-Seg{k + 1}.mat'
                # Load MATLAB .mat file
                mat_contents = loadmat(name)
                V = mat_contents['V']  # (89928, 500)
                U = mat_contents['U']  # (500, 4800)

                VU = V[start_V:end_V, :].dot(U)  # (T, 4800)
                seg = VU.reshape((used_V, Wid0, Hei0))
                Wid1, Hei1 = Wid0 // resize_factor, Hei0 // resize_factor
                seg = resize(seg, (used_V, Wid1, Hei1), mode='constant')

                i, j = k // 4, (k % 4)
                Data[:, i * Wid1:(i+1) * Wid1, j * Hei1:(j+1) * Hei1] = seg

            #### dimension reduce video
            t = np.arange(Data.shape[0]) / video_sampling_rate
            d = np.array(Data.reshape(Data.shape[0], -1))
            del Data
            d = proSVD.apply_and_cache(input_arr=d, output_d=video_target_dim, init_size=video_target_dim)
            t, d = clip(t, d)

            #### define times
            ca_times = np.hstack([np.linspace(*trial_edges[i], n_samples_per_trial) for i in range(len(trial_edges))])
            ca_times = ca_times / video_sampling_rate

            return A, d, ca_times, t

        return static_construct(self.cam, self.video_target_dim, self.resize_factor)

    def acquire(self):
        if not self.inner_data_path.is_dir():
            # TODO: I think this is actually publicly downloadable
            print(f"""\
Please ask Anne Draelos where to download the Musal data.\
""")
            raise FileNotFoundError()


class Naumann24uDataset(Dataset):
    doi = None
    automatically_downloadable = False
    model_organism = ModelOrganism.FISH
    dataset_base_path = DATA_BASE_PATH / "naumann24u"
    sub_datasets = (
        "output_020424_ds1",
        "output_012824_ds3",
        "output_012824_ds6_fish3",
    )

    def __init__(self, sub_dataset_identifier=sub_datasets[0]):
        self.sub_dataset = sub_dataset_identifier
        self.C, self.opto_stimulations, self.neuron_df, self.visual_stimuli = self.construct(sub_dataset_identifier)
        self.neural_data = NumpyTimedDataSource(self.C.T, np.arange(self.C.shape[1]))
        self.behavioral_data = None

    def construct(self, sub_dataset_identifier):
        visual_stimuli, optical_stimulations, C = self.acquire(sub_dataset_identifier)

        visual_stimuli_df = pd.DataFrame({'sample': visual_stimuli[:,0], 'l_angle': visual_stimuli[:,2], 'r_angle': visual_stimuli[:,3]})

        optical_stimulation_df = pd.DataFrame({'sample': optical_stimulations[:, 0], 'target_neuron': optical_stimulations[:,2]})

        neurons = {}
        for neuron_id in optical_stimulation_df['target_neuron']:
            locations = optical_stimulations[optical_stimulations[:,2] == neuron_id, 3:]
            assert np.all(np.std(locations, axis=0) == 0)
            neurons[neuron_id] = locations[0,:]
        neuron_df = pd.DataFrame.from_dict(neurons, orient='index', columns=['x', 'y'])


        return C, optical_stimulation_df, neuron_df, visual_stimuli_df

    def acquire(self, sub_dataset_identifier):
        base = self.dataset_base_path / sub_dataset_identifier
        if not base.is_dir():
            print(base)
            print("""\
Please ask Anne Draelos how to acquire the Naumann lab dataset we use here. (hint: box)\
""")
            raise FileNotFoundError()
        optical_stimulations = np.load(base/'photostims.npy')
        visual_stimuli = np.loadtxt(base/'stimmed.txt')

        c_filename = 'raw_C.txt'
        if sub_dataset_identifier == 'output_020424_ds1':
            c_filename = 'analysis_proc_C.txt'
        C = np.loadtxt(base/c_filename)

        return visual_stimuli, optical_stimulations, C

    def plot_colors(self, ax):
        theta = np.linspace(0, 360)
        ax.scatter(np.cos(theta * np.pi / 180), np.sin(theta * np.pi / 180), c=self.a2c(theta))
        ax.axis('equal')

    @staticmethod
    def a2c(a):
        a = (a + 30) % 360
        return matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(vmin=0, vmax=360), cmap=matplotlib.cm.hsv).to_rgba(a)


class Leventhal24uDataset(Dataset):
    doi = None
    model_organism = ModelOrganism.RAT
    dataset_base_path = DATA_BASE_PATH / 'leventhal24u'
    automatically_downloadable = False
#
    sub_datasets = (
        "R0493/R0493_20230720_ChAdvanced_230720_105441",
        "R0466/R0466_20230403_ChoiceEasy_230403_095044",
        # "R0544_20240625_ChoiceEasy_240625_100738"
    )
#
    def __init__(self, sub_dataset_identifier=sub_datasets[0], bin_size=0.03):
        self.sub_dataset = sub_dataset_identifier
        self.bin_size = bin_size
        A, t, spike_times, clusters, trial_data, unflattened_trial_data = self.construct(sub_dataset_identifier)
        self.neural_data = NumpyTimedDataSource(A, t)
        self.spike_times = spike_times
        self.spike_clusters = clusters
        self.trial_data = trial_data
        self.unflattened_trial_data = unflattened_trial_data
        # self.behavioral_data = NumpyTimedDataSource(beh,t)
#
    def construct(self, sub_dataset_identifier):
        spike_times, clusters, trials = self.acquire(sub_dataset_identifier)

        # for neuron_to_drop in [846]:
        #     warnings.warn(f"dropping neuron {neuron_to_drop} because I suspect it was an 'other' unit; I need to check this")
        #     s = clusters != neuron_to_drop
        #     spike_times = spike_times[s]
        #     clusters = clusters[s]

        unique_clusters = np.unique(clusters)
        n_units = unique_clusters.size

        n_bins = np.ceil((spike_times.max() - spike_times.min()) / self.bin_size).astype(int) + 1
        bin_edges = np.arange(n_bins)*self.bin_size + spike_times.min()

        A = np.zeros((n_bins - 1, n_units))
        for i, c in enumerate(unique_clusters):
            A[:, i] = np.histogram(spike_times[clusters == c], bins=bin_edges)[0]

        bin_ends = bin_edges[1:]

        trial_data = pd.DataFrame(trials['trials'])

        df = trial_data

        df = df.rename(columns={'tone': 'toneType'})
        for column in ['timing', 'timestamps']:
            sub_df = df[column].apply(pd.Series)
            if column == 'timing':
                sub_df = sub_df.rename(columns=lambda x: "relative_" + x)

            df = pd.concat([df.drop(column, axis=1), sub_df], axis=1)
        flattened_trial_data = df

        # expected_trial_data_keys = {'Time', 'Attempt', 'Center', 'Target', 'Tone', 'RT', 'MT', 'pretone', 'outcome', 'SideNP', 'CenterNoseIn', 'SideInToFood'}
        # discovered_trial_data_keys = {k for k, v in log_data.items() if hasattr(v, '__len__') and len(v) == len(log_data['Time'])}
        # assert expected_trial_data_keys.difference(discovered_trial_data_keys) == set()
        # assert discovered_trial_data_keys.difference(expected_trial_data_keys) == set()
        #
        # trial_data = pd.DataFrame({key: log_data[key] for key in expected_trial_data_keys})

        return A, bin_ends, spike_times, clusters, flattened_trial_data, trial_data
#
    def acquire(self, sub_dataset_identifier):
        subset_base_path = self.dataset_base_path / sub_dataset_identifier

        lib_directory = (self.dataset_base_path / 'load-rhd-notebook-python').resolve()
        if not lib_directory.is_dir():
            print(f"""\
Please download (clone) `https://github.com/Intan-Technologies/load-rhd-notebook-python` into {self.dataset_base_path.resolve()}.""")

        info_file = subset_base_path / 'info.rhd'
        if not info_file.is_file():
            print(f"""\
Please place a copy of '{sub_dataset_identifier}' into '{self.dataset_base_path}'.""")

        sys.path.append(str(lib_directory)) # todo: this is really bad
        import importrhdutilities as rhd

        result, data_present = rhd.load_file(subset_base_path / 'info.rhd')
        assert not data_present
        sampling_frequency = result['frequency_parameters']['amplifier_sample_rate']

        spike_times = np.load(subset_base_path / 'spike_times.npy').flatten() / sampling_frequency
        clusters = np.load(subset_base_path / 'spike_clusters.npy').flatten()

        trials = scipy.io.loadmat(subset_base_path.parent / 'trials.mat', squeeze_me=True, simplify_cells=True)

        return spike_times, clusters, trials

        # t = np.fromfile(subset_base_path/'time.dat', dtype='int32') / sampling_frequency


class Zong22Dataset(Dataset):
    doi = "https://dx.doi.org/10.11582/2022.00008"
    automatically_downloadable = False
    model_organism = ModelOrganism.MOUSE
    dataset_base_path = DATA_BASE_PATH / 'zong22'

    def make_cookie_entry(area, animal_id, date, f_part, f_total, cookie_status, filtered):
        assert type(area) == str  # this can become static after 3.10
        cookie_status = 'with' if cookie_status else 'no'
        filtered = 'filtered' if filtered else ''
        return {
            'basepath':       f'{area}_recordings/{animal_id}/{date}/',
            'raw_frames':     f'{animal_id}_imaging_{date}_{cookie_status}cookies_00001.tif',
            'behavior_csv':   f'{animal_id}_imaging_{date}_{cookie_status}cookies_00001_trackingVideoDLC_resnet50_OPENMINI2P_bottomcameraAug26shuffle1_1030000{filtered}.csv',
            'behavior_video': f'{animal_id}_imaging_{date}_{cookie_status}cookies_00001_trackingVideo.avi',
            'part_of_F': (f_part,f_total)
        }

    def make_object_entry(area, animal_id, date, f_part, f_total, object_n, filtered):
        assert type(area) == str  # this can become static after 3.10
        object_str = f'object{object_n}' if object_n is not None else 'noobject'
        filtered = 'filtered' if filtered else ''
        return {
            'basepath':       f'{area}_recordings/{animal_id}/{date}/',
            'raw_frames':     f'{animal_id}_imaging_{date}_{object_str}_00001.tif',
            'behavior_csv':   f'{animal_id}_imaging_{date}_{object_str}_00001_trackingVideoDLC_resnet50_OPENMINI2P_bottomcameraAug26shuffle1_1030000{filtered}.csv',
            # 'behavior_video': f'{animal_id}_imaging_{date}_{object_str}_00001_trackingVideo.avi',
            'part_of_F': (f_part,f_total)
        }

    sub_datset_info = pd.DataFrame([
        make_cookie_entry('VC', '93562', '20200817', 1, 2, False, True),
        make_cookie_entry('VC', '93562', '20200817', 2, 2, True, True),

        make_cookie_entry('MEC', '94557', '20200822', 1, 2, False, False),
        make_cookie_entry('MEC', '94557', '20200822', 1, 2, True, False),

        make_object_entry('MEC', '94557', '20201008', 1, 3, None, True),
        make_object_entry('MEC', '94557', '20201008', 2, 3, 1, True),
        make_object_entry('MEC', '94557', '20201008', 3, 3, 2, True),

        {
            'basepath': 'CA1_recordings/97288/20210315/',
            'raw_frames': '97288_20210315_00002.tif',
            'part_of_F': (1,1)
        },
    ])

    sub_datasets = list(sub_datset_info.index)

    def __init__(self, sub_dataset_identifier=sub_datasets[0]):
        self.sub_dataset = sub_dataset_identifier
        self.neural_Fs = 15
        self.bin_width = 1/self.neural_Fs  # todo: make this universal?
        self.F, self.raw_images, self.behavior_video, self.behavior_df, self.n_cells, self.stat, self.ops = self.acquire()

        self.neural_data = NumpyTimedDataSource(self.F.T, np.arange(self.F.shape[1]) * 1 / self.neural_Fs)
        self.behavioral_data = NumpyTimedDataSource(timepoints=self.behavior_df.loc[:, 't'], source=self.behavior_df.loc[:, ['x', 'y', 'hd', 'h2b']])

        self.video_t = np.squeeze(self.behavioral_data.t)

    def acquire(self):
        sub_dataset_base_path = self.dataset_base_path / self.sub_datset_info.basepath[self.sub_dataset]
        if not sub_dataset_base_path.is_dir():
            print(f"Go download the dataset from {self.doi}. (Or remount the external drive on Tycho)")
            raise FileNotFoundError()

        iscell = np.load(sub_dataset_base_path / 'suite2p' / 'plane0' / 'iscell.npy')
        F_all = np.load(sub_dataset_base_path / 'suite2p' / 'plane0' / 'F.npy')
        n_cells = int(sum(iscell[:, 0]))

        stat = np.load(sub_dataset_base_path / 'suite2p' / 'plane0' / 'stat.npy', allow_pickle=True)
        ops = np.load(sub_dataset_base_path / 'suite2p' / 'plane0' / 'ops.npy', allow_pickle=True).item()

        def make_beh(fpath):
            pre_beh = pd.read_csv(fpath)
            columns = ["t"] + list(map(lambda a: f"{a[0]}_{a[1]}", zip(pre_beh.iloc[0, 1:], pre_beh.iloc[1, 1:])))
            columns = {pre_beh.columns[i]: columns[i] for i in range(len(columns))}
            beh = pre_beh.rename(columns=columns).iloc[2:].astype(float).reset_index(drop=True)
            beh.t = beh.t / self.neural_Fs
            return beh

        part, total = self.sub_datset_info.part_of_F[self.sub_dataset]
        block_length = F_all.shape[1] // total
        F = F_all[:, (part - 1) * block_length: part * block_length]
        img = Image.open(sub_dataset_base_path / self.sub_datset_info.raw_frames[self.sub_dataset])
        video = pims.Video(sub_dataset_base_path / self.sub_datset_info.behavior_video[self.sub_dataset])
        beh = make_beh(sub_dataset_base_path / self.sub_datset_info.behavior_csv[self.sub_dataset])

        nose = self.get_behavior_trace(beh, 'nose')
        body = self.get_behavior_trace(beh, 'bodycenter')
        head = self.get_behavior_trace(beh, 'mouse')

        beh['hd'] = np.arctan2(*(nose - head).T)
        beh['h2b'] = np.linalg.norm(head - body, axis=1)
        beh['x'] = head[:,0]
        beh['y'] = head[:,1]


        return F, img, video, beh, n_cells, stat, ops

    def show_stim_pattern(self, ax, desired_stim):
        planes = []
        for i in range(500):
            self.raw_images.seek(i)
            planes.append(np.array(self.raw_images))

        im = np.mean(planes, axis=0)

        ax.matshow(-im, cmap='Grays')
        xs, ys = list(zip(*[cell['med'] for cell in self.stat]))
        map = ax.scatter(ys, xs, s=7, c=desired_stim)

        ax.get_figure().colorbar(map)

    @staticmethod
    def get_behavior_trace(beh, point_str, threshold=.999):
        point_trace = np.array([beh.loc[:, point_str + '_x'].to_numpy(),
                                beh.loc[:, point_str + '_y'].to_numpy()]).T
        s = beh.loc[:, point_str + '_likelihood'].to_numpy() < threshold
        point_trace[s] *= np.nan
        return point_trace


class DummyCircleDataset(Dataset):
    # meant to mock Odoherty21
    doi = None
    automatically_downloadable = False
    model_organism = None

    def __init__(self, Fs=33, total_time=600):
        self.rng = np.random.default_rng()
        self.Fs = Fs
        self.bin_width = 1 / self.Fs
        self.t = np.linspace(0, total_time, total_time * self.Fs + 1)

        neural_data, behavioral_data = self.construct()

        self.neural_data = NumpyTimedDataSource(neural_data, self.t)
        self.behavioral_data = NumpyTimedDataSource(behavioral_data, self.t)

    def acquire(self):
        pass

    def construct(self):
        self.acquire()
        # 15 samples is about 2pi end_time
        # 1 sample is bin_width start_time
        speed_factor = self.Fs * np.pi * 2 / 15
        neural_data = np.sin(np.vstack(3 * [self.t]).T * speed_factor)
        neural_data = neural_data + self.rng.normal(scale=.1, size=neural_data.shape)

        behavioral_data = np.sin(np.vstack(3 * [self.t]).T * speed_factor)
        behavioral_data = behavioral_data + self.rng.normal(scale=.1, size=behavioral_data.shape)
        return neural_data, behavioral_data


"""
class Low21Dataset(MultiDataset):
    doi = "https://doi.org/10.17632/hntn6m2pgk.1"
    automatically_downloadable = True
    dataset_base_path = DATA_BASE_PATH / 'low21'

    def acquire(self, sub_dataset_identifier=None):
        if len(list(self.dataset_base_path.glob("*.npy"))) == 0:
            datahugger.get(self.doi, self.dataset_base_path)

    # def construct(self, sub_dataset_identifier=None):
    #     pass
    #
    # def get_sub_datasets(self):
    #     pass
"""


