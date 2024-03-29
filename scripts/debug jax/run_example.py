import os
import sys

match sys.argv[1]:
    case "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    case "gpu":
        pass
    case _:
        raise Exception()

from adaptive_latents import Bubblewrap, BWRun, AnimationManager, NumpyTimedDataSource, default_rwd_parameters
import adaptive_latents.plotting_functions as pfs
from adaptive_latents import CONFIG
import numpy as np

def main():
    t = np.arange(0, 2*np.pi*50, np.pi/5)
    X = np.column_stack([np.cos(t), np.sin(t)])
    X = X + np.random.default_rng(0).normal(size=X.shape)*1e-3
    in_ds = NumpyTimedDataSource(X, t, time_offsets=(0,1))
    

    bw = Bubblewrap(dim=2, **dict(default_rwd_parameters, num=200, num_grad_q=1, step=8e-2))

    # define the object to coordinate all the other objects
    br = BWRun(bw=bw, in_ds=in_ds, show_tqdm=True, log_level=100)

    br.run(save=True)
    # print(f"{numpy.array(bw.L)[0][0][0]:.32f} ({ xla_bridge.get_backend().platform })")

if __name__ == '__main__':
    main()




