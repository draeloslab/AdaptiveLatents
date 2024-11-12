from adaptive_latents import (
    datasets,
    Concatenator,
    Pipeline,
    CenteringTransformer,
    KernelSmoother,
    proSVD,
    Bubblewrap,
    VanillaOnlineRegressor,
    sjPCA,
    mmICA,
)
from adaptive_latents.timed_data_source import ArrayWithTime, NumpyTimedDataSource
from adaptive_latents.utils import evaluate_regression
from adaptive_latents.transformer import Tee
import numpy as np
import functools
import copy
import matplotlib.pyplot as plt
from adaptive_latents.transformer import PassThroughDict

class PredictionEvaluation:
    def __init__(self, sources, pipeline=None, target_pairs=None, exit_time=None, stream_names=None):
        self.target_pairs = target_pairs or {'default': (2, 1)}  # default assumes stream 0 is x's, stream 1 is y's, and stream 2 is query x's
        self.pipeline = pipeline or Pipeline()
        self.sources = sources
        self.target_pairs = target_pairs
        self.stream_names = stream_names or PassThroughDict()

        outputs = pipeline.offline_run_on(sources, convinient_return=False, exit_time=exit_time)

        self.outputs = {}
        for k, v in outputs.items():
            self.outputs[k] = ArrayWithTime.from_list(v, drop_early_nans=True, squeeze_type='to_2d')
            if k in self.stream_names:
                self.outputs[self.stream_names[k]] = self.outputs[k]

        self.evaluations = {}
        for name, (estimate_stream, target_stream) in self.target_pairs.items():
            estimate = self.outputs[estimate_stream]
            target = self.outputs[target_stream]
            self.evaluations[name] = evaluate_regression(estimate, estimate.t, target, target.t)

        if len(self.evaluations) == 1:
            self.evaluation = self.evaluations[list(self.evaluations.keys())[0]]

def pred_reg_run(
        neural_data,
        behavioral_data,
        target_data,
        n_bubbles=875,
        bw_step=10 ** 0.25,
        neural_smoothing_tau=.688,
        stream_scaling_factors=None,
        neural_lag=0,
        exit_time=None,
        dim_red_method='pro',
        log_level=1,
        **kwargs,
):

    stream_scaling_factors = stream_scaling_factors or {0: 1, 1: 0}

    neural_data = copy.deepcopy(neural_data)
    neural_data.t = neural_data.t + neural_lag

    sources = [
        (neural_data, 0),
        (behavioral_data, 1),
        (target_data, 2),
        (NumpyTimedDataSource(neural_data.t[1:] - neural_data.t[:-1], neural_data.t[:-1]), 3)
    ]

    bw = functools.partial(
        Bubblewrap,
        num=n_bubbles,
        M=500,
        lam=1e-3,
        nu=1e-3,
        eps=1e-4,
        step=bw_step,
        num_grad_q=1,
        sigma_orig_adjustment=100,
        log_level=log_level,
        check_consistent_dt=False,
    )

    dim_red = {
        'pro': Pipeline(log_level=log_level),
        'sjpca': sjPCA(log_level=log_level),
        'mmica': mmICA(log_level=log_level),
    }.get(dim_red_method)

    pipeline = Pipeline([
        CenteringTransformer(init_size=100, log_level=log_level),
        KernelSmoother(tau=neural_smoothing_tau / neural_data.dt, log_level=log_level),
        Concatenator(input_streams={0: 0, 1: 1}, output_streams={0: 0, 1: 0, 'skip': -1}, stream_scaling_factors=stream_scaling_factors, log_level=log_level),
        proSVD(k=6, log_level=log_level),
        dim_red,
        tee:=Tee(input_streams={0:0}, log_level=log_level),
        bw(input_streams={0: 'X', 3: 'dt'}, log_level=log_level),
        VanillaOnlineRegressor(input_streams={0: 'X', 2: 'Y', 3: 'qX'}, log_level=log_level)
    ], log_level=log_level)

    e = PredictionEvaluation(sources, pipeline, target_pairs={'joint to beh': (3,2)}, stream_names={3:'pred', 2:'target'}, exit_time=exit_time)
    tee.convert_to_array()
    e.dim_reduced_data = tee.observed[0]
    return e


def pred_reg_run_with_defaults(ds_name, **kwargs):
    if ds_name == 'odoherty21':
        args = dict(
            neural_smoothing_tau=.12,
            bw_step=10 ** -1.5,
            n_bubbles=1100,
            stream_scaling_factors={0: 1, 1: 1},
            drop_third_coord=True,
            exit_time=None,
        )

        args |= kwargs

        d = datasets.Odoherty21Dataset(drop_third_coord=args['drop_third_coord'])
        neural_data = d.neural_data
        behavioral_data = d.behavioral_data
    elif ds_name == 'zong22':
        args = dict(
            n_bubbles=875,
            bw_step=10 ** 0.25,
            neural_smoothing_tau=.688,
            sub_dataset_identifier=2,
            pos_scale=1 / 160, hd_scale=1 / 1.8, h2b_scale=1 / 8.5,
            stream_scaling_factors={0: 1 / 1000 * 10 ** -1, 1: 0},
            exit_time=None,
        )
        args |= kwargs

        d = datasets.Zong22Dataset(sub_dataset_identifier=args['sub_dataset_identifier'], pos_scale=args['pos_scale'],
                                   hd_scale=args['hd_scale'], h2b_scale=args['h2b_scale'])
        neural_data = d.neural_data
        behavioral_data = d.behavioral_data
    elif ds_name == 'naumann24u':
        args = dict(
            n_bubbles=875,
            bw_step=10 ** 0.25,
            neural_smoothing_tau=.688,
            sub_dataset_identifier=datasets.Naumann24uDataset.sub_datasets[1],
            stream_scaling_factors={0: 1, 1: 0},
            beh_type='angle',
            exit_time=None,
        )
        args |= kwargs

        d = datasets.Naumann24uDataset(sub_dataset_identifier=args['sub_dataset_identifier'], beh_type=args['beh_type'])
        neural_data = d.neural_data
        behavioral_data = d.behavioral_data
    else:
        raise ValueError()

    result = pred_reg_run(neural_data=neural_data, behavioral_data=behavioral_data, target_data=behavioral_data, **args)
    result.dataset = ds_name
    return result






def plot_flow_fields(dim_reduced_data, x_direction=0, y_direction=1, grid_n=13, scatter_alpha=0, normalize_method=None):
    assert normalize_method in {None, 'none', 'diffs', 'hcubes', 'squares'}
    fig, axs = plt.subplots(nrows=1, ncols=len(dim_reduced_data), squeeze=False, layout='tight', figsize=(12,4))

    for idx, (name, latents) in enumerate(dim_reduced_data.items()):
        e1, e2 = np.zeros(latents.shape[1]), np.zeros(latents.shape[1])
        e1[x_direction] = 1
        e2[y_direction] = 1

        ax: plt.Axes = axs[0, idx]
        ax.scatter(latents @ e1, latents @ e2, s=5, alpha=scatter_alpha)
        x1, x2, y1, y2 = ax.axis()
        x_points = np.linspace(x1, x2, grid_n)
        y_points = np.linspace(y1, y2, grid_n)

        d_latents = np.diff(latents, axis=0)
        if normalize_method == 'diffs':
            d_latents = d_latents / np.linalg.norm(d_latents, axis=1)[:, np.newaxis]


        origins = []
        arrows = []
        n_points = []
        for i in range(len(x_points) - 1):
            for j in range(len(y_points) - 1):
                proj_1 = (latents[:-1] @ e1)
                proj_2 = (latents[:-1] @ e2)
                # s stands for slice
                s = (
                        (x_points[i] <= proj_1) & (proj_1 < x_points[i + 1])
                        &
                        (y_points[j] <= proj_2) & (proj_2 < y_points[j + 1])
                )
                if s.sum():
                    arrow = np.nanmean(d_latents[s],axis=0)
                    if normalize_method == 'hcubes':
                        arrow = arrow / np.linalg.norm(arrow)
                    arrows.append(arrow)
                    origins.append([np.nanmean(x_points[i:i + 2]), np.nanmean(y_points[j:j + 2])])
                    n_points.append(s.sum())
        origins, arrows, n_points = np.array(origins), np.array(arrows), np.array(n_points)
        arrows = np.array([arrows @ e1, arrows @ e2]).T
        if normalize_method == 'squares':
            arrows = arrows / np.linalg.norm(arrows, axis=1)[:, np.newaxis]

        ax.quiver(origins[:, 0], origins[:, 1], arrows[:,0], arrows[:,1], scale=1 / 20, units='dots', color='red')

        ax.axis('equal')
        ax.axis('off')


if __name__ == '__main__':
    odoherty_run = pred_reg_run_with_defaults(ds_name='odoherty21', exit_time=60)