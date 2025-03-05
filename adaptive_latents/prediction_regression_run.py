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
from adaptive_latents.timed_data_source import ArrayWithTime
from adaptive_latents.utils import evaluate_regression
from adaptive_latents.transformer import Tee
import numpy as np
import functools
import copy
import matplotlib.pyplot as plt
from adaptive_latents.transformer import PassThroughDict

class PredictionEvaluation:
    def __init__(self, sources, pipeline=None, target_pairs=None, exit_time=None, stream_names=None, evaluate=True):
        self.target_pairs = target_pairs or {'default': (2, 1)}  # default assumes stream 0 is x's, stream 1 is y's, and stream 2 is query x's
        self.pipeline = pipeline or Pipeline()
        self.sources = sources
        self.target_pairs = target_pairs
        self.stream_names = stream_names or PassThroughDict()
        self.exit_time = exit_time


        self.dim_reduced_data: ArrayWithTime | None = None  # to be set by another object
        self.outputs = {}
        self.evaluations = {}
        if evaluate:
            self.evaluate()

    def evaluate(self, outputs=None):
        if outputs is None :
            outputs = self.pipeline.offline_run_on(self.sources, convinient_return=False, exit_time=self.exit_time)
        self.outputs = outputs

        for k, v in dict(self.outputs).items():
            self.outputs[k] = ArrayWithTime.from_list(v, drop_early_nans=True, squeeze_type='to_2d')
            if k in self.stream_names:
                self.outputs[self.stream_names[k]] = self.outputs[k]

        for name, (estimate_stream, target_stream) in self.target_pairs.items():
            estimate = self.outputs[estimate_stream]
            target = self.outputs[target_stream]
            self.evaluations[name] = evaluate_regression(estimate, estimate.t, target, target.t)


def pred_reg_run(
        neural_data,
        behavioral_data,
        target_data,
        predict=False,
        n_bubbles=875,
        bw_step=10 ** 0.25,
        neural_smoothing_tau=.688,
        stream_scaling_factors=None,
        neural_lag=0,
        exit_time=None,
        dim_red_method='pro',
        log_level=1,
        evaluate=True,
        **kwargs,
):
    stream_scaling_factors = stream_scaling_factors or {0: 1, 1: 0}

    neural_data = copy.deepcopy(neural_data)
    neural_data.t = neural_data.t + neural_lag

    sources = [
        (neural_data, 0),
        (behavioral_data, 1),
        (target_data, 2),
        (ArrayWithTime((neural_data.t[1:] - neural_data.t[:-1]).reshape(-1, 1, 1), neural_data.t[:-1]), 3)
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

    last_steps = [bw(input_streams={0: 'X', 3: 'dt'}, log_level=log_level),
                  VanillaOnlineRegressor(input_streams={0: 'X', 2: 'Y', 3: 'qX'},
                                         log_level=log_level)] if predict else []

    pipeline = Pipeline([
        CenteringTransformer(init_size=100, log_level=log_level),
        KernelSmoother(tau=neural_smoothing_tau / neural_data.dt, log_level=log_level),
        Concatenator(input_streams={0: 0, 1: 1}, output_streams={0: 0, 1: 0, 'skip': -1}, stream_scaling_factors=stream_scaling_factors, log_level=log_level),
        proSVD(k=6, log_level=log_level),
        dim_red,
        tee := Tee(input_streams={0: 0}, log_level=log_level),
        *last_steps
    ], log_level=log_level)

    e = PredictionEvaluation(sources, pipeline, target_pairs={'joint to beh': (3, 2)} if predict else {}, stream_names={3: 'pred', 2: 'target'}, exit_time=exit_time, evaluate=evaluate)

    if evaluate:
        tee.convert_to_array()
        e.dim_reduced_data = tee.observed[0]
    return e

defaults_per_dataset = {
    'odoherty21': dict(
        neural_smoothing_tau=.12,
        bw_step=10 ** -1.5,
        n_bubbles=1100,
        stream_scaling_factors={0: 1, 1: 1},
        drop_third_coord=True,
        exit_time=None,
    ),

    'zong22': dict(
        n_bubbles=875,
        bw_step=10 ** 0.25,
        neural_smoothing_tau=.688,
        sub_dataset_identifier=2,
        pos_scale=1 / 160, hd_scale=1 / 1.8, h2b_scale=1 / 8.5,
        stream_scaling_factors={0: 1 / 1000 * 10 ** -1, 1: 0},
        exit_time=None,
    ),

    'naumann24u': dict(
        n_bubbles=875,
        bw_step=10 ** 0.25,
        neural_smoothing_tau=.688,
        sub_dataset_identifier=datasets.Naumann24uDataset.sub_datasets[1],
        stream_scaling_factors={0: 1, 1: 0},
        beh_type='angle',
        exit_time=None,
    )

}
def pred_reg_run_with_defaults(ds_name, **kwargs):
    """
    Examples
    -------
    >>> pred_reg_run_with_defaults('odoherty21', exit_time=60)
    <...PredictionEvaluation...>
    """

    if ds_name == 'odoherty21':
        args = dict(defaults_per_dataset[ds_name])
        args |= kwargs

        d = datasets.Odoherty21Dataset(drop_third_coord=args['drop_third_coord'])
        neural_data = d.neural_data
        behavioral_data = d.behavioral_data
    elif ds_name == 'zong22':
        args = dict(defaults_per_dataset[ds_name])
        args |= kwargs

        d = datasets.Zong22Dataset(sub_dataset_identifier=args['sub_dataset_identifier'], pos_scale=args['pos_scale'],
                                   hd_scale=args['hd_scale'], h2b_scale=args['h2b_scale'])
        neural_data = d.neural_data
        behavioral_data = d.behavioral_data
    elif ds_name == 'naumann24u':
        args = dict(defaults_per_dataset[ds_name])
        args |= kwargs

        d = datasets.Naumann24uDataset(sub_dataset_identifier=args['sub_dataset_identifier'], beh_type=args['beh_type'])
        neural_data = d.neural_data
        behavioral_data = d.behavioral_data
    else:
        raise ValueError()

    result = pred_reg_run(neural_data=neural_data, behavioral_data=behavioral_data, target_data=behavioral_data, **args)
    result.dataset = ds_name
    return result


