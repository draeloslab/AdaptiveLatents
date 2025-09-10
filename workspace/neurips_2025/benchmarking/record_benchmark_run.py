import copy
from collections import deque
import jax

from adaptive_latents import StreamingKalmanFilter, ArrayWithTime, Pipeline, StimRegressor, Bubblewrap, proSVD, CenteringTransformer, VJF, KernelSmoother, mmICA, sjPCA, datasets
from adaptive_latents.regressions import BaseKernelRegressor
import numpy as np
from adaptive_latents.stim_designer import StimDesigner
import functools

from types import SimpleNamespace
import time

from adaptive_latents.sim_stim import make_sr


if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    parser.add_argument("--pred_type", type=str, required=True, choices=['kf', 'bw', 'vjf'])
    parser.add_argument("--dimred_type", type=str, required=True, choices=['prosvd', 'sjpca', 'mmica'])
    args = parser.parse_args()



    rng = np.random.default_rng(0) # sfc64?
    neural_data = datasets.Odoherty21Dataset().neural_data

    autoreg = None
    match args.pred_type:
        case "kf":
            autoreg = functools.partial(StreamingKalmanFilter)
        case "bw":
            autoreg = functools.partial(Bubblewrap)
        case "vjf":
            autoreg = functools.partial(VJF)

    sr, stim_designer, log = make_sr(
        neural_data,
        rng,
        autoreg=autoreg,
        exit_time=np.nan,
        design_method='optimized learned u_to_s',
        last_dim_red=args.dimred_type,
        stim_reg_maxlen=200,
    )
    print(f"{log['timing_log'].init_time = }")
    print(f"{log['timing_log'].loop_time = }")

    import pandas as pd

    df = pd.DataFrame({k:v for k,v in log['timing_log'].__dict__.items() if isinstance(v, list)})
    df.to_csv(args.output, index=False)