import numpy as np
from adaptive_latents import Bubblewrap, BWRun, AnimationManager
from adaptive_latents.regressions import NearestNeighborRegressor, SymmetricNoisyRegressor
from adaptive_latents.default_parameters import default_jpca_dataset_parameters
from adaptive_latents.input_sources.functional import get_from_saved_npz
from adaptive_latents.input_sources.data_sources import NumpyTimedDataSource
import adaptive_latents.plotting_functions as bpf
from adaptive_latents import CONFIG

def main(output_directory=CONFIG["output_path"]/"bubblewrap_runs", steps_to_run=None):
    obs, beh = get_from_saved_npz("jpca_reduced_sc.npz")
    in_ds = NumpyTimedDataSource(obs, None, time_offsets=(1,))
    out_ds = NumpyTimedDataSource(beh, None, time_offsets=(1,))

    # define the adaptive_latents object
    bw = Bubblewrap(dim=in_ds.output_shape,  **dict(default_jpca_dataset_parameters, B_thresh=-15, copy_row_on_teleport=False))

    # define the (optional) method to regress the HMM state from `bw.alpha`
    # reg = SymmetricNoisyRegressor(input_d=bw.N, output_d=1)
    reg = NearestNeighborRegressor(input_d=bw.N, output_d=1, maxlen=600)

    class CustomAnimation(AnimationManager):
        n_rows = 1
        n_cols = 1
        figsize = (15,10)
        extension = "mp4"

        def custom_draw_frame(self, step, bw: Bubblewrap, br: BWRun):
            historical_observations = br.obs_ds.get_history()

            bpf.show_bubbles_2d(self.ax[0,0], historical_observations, bw, alpha_coefficient=.5)
            self.ax[0,0].set_title(f"Step {step}")
        def frame_draw_condition(self, step_number, bw):
            return step_number % 5 == 0
    am = CustomAnimation()

    # define the object to coordinate all the other objects
    br = BWRun(bw=bw, obs_ds=in_ds, beh_ds=out_ds, behavior_regressor=reg, animation_manager=am, show_tqdm=True, output_directory=output_directory)

    # run and save the output
    br.run(limit=steps_to_run)

if __name__ == '__main__':
    main(steps_to_run=100)
