import os
import sys

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
    
    t = np.arange(0, 2*np.pi*200, np.pi/5)
    X = np.column_stack([np.cos(t), np.sin(t)])
    X = X + np.random.default_rng(0).normal(size=X.shape)*1e-3
    in_ds = NumpyTimedDataSource(X, np.arange(t.shape[0]), time_offsets=(0,1))

    bw = Bubblewrap(dim=X.shape[1], **dict(default_rwd_parameters, num=200, num_grad_q=1, step=8e-2))

    class CustomAnimation(AnimationManager):
        n_rows = 2
        n_cols = 2
        figsize = (15,10)
        extension = "mp4"

        def custom_draw_frame(self, step, bw: Bubblewrap, br: BWRun):
            historical_observations, _ = br.obs_ds.get_history()

            # pfs.show_bubbles_2d(self.ax[0,1], historical_observations, bw, alpha_coefficient=.5)
            pfs.show_B(self.ax[0,1], br, show_log=False)

            pfs.show_A(self.ax[0,0], self.fig, bw, show_log=True)
            self.ax[0,0].set_title(f"log A (Step {step})")

            pfs.show_alpha(self.ax[1,0], br, offset=1, show_log=True)

            pfs.show_A_eigenspectrum(self.ax[1,1], bw)


        def frame_draw_condition(self, step_number, bw):
            return 150 < step_number < 300

    am = None
    # am = CustomAnimation()

    # define the object to coordinate all the other objects
    br = BWRun(bw=bw, obs_ds=in_ds, animation_manager=am, show_tqdm=False, save_A=True)

    br.run(limit=limit, save=True, freeze=False if limit is not None else True)
    # print(f"{numpy.array(bw.L)[0][0][0]:.32f} ({ xla_bridge.get_backend().platform })")

if __name__ == '__main__':
    main()




