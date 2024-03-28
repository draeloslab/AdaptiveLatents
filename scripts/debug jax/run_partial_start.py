import os
import sys
from jax.config import config
config.update("jax_enable_x64", True)

backend = sys.argv[1]
match backend:
    case "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    case "gpu":
        pass
    case _:
        raise Exception()

limit = eval(sys.argv[2])

from adaptive_latents import Bubblewrap, BWRun, AnimationManager, NumpyTimedDataSource, default_rwd_parameters
import adaptive_latents.plotting_functions as pfs
from adaptive_latents import CONFIG
import numpy as np

def main():
    
    t = np.arange(0, 2*np.pi*100, np.pi/5)
    X = np.column_stack([np.cos(t), np.sin(t)])
    X = X + np.random.default_rng(0).normal(size=X.shape)*1e-3
    in_ds = NumpyTimedDataSource(X, np.arange(t.shape[0]), time_offsets=(0,1))

    bw = Bubblewrap(dim=X.shape[1], **dict(default_rwd_parameters, num=200, num_grad_q=1, step=8e-2))


    # define the object to coordinate all the other objects
    br = BWRun(bw=bw, in_ds=in_ds, show_tqdm=False, save_A=True)

    br.run(limit=limit, save=True, freeze=False if limit is not None else True)

if __name__ == '__main__':
    main()




