from adaptive_latents import Bubblewrap, BWRun, AnimationManager, SemiRegularizedRegressor, NumpyTimedDataSource, default_rwd_parameters
from adaptive_latents.input_sources.hmm_simulation import simulate_example_data
import adaptive_latents.plotting_functions as pfs
from adaptive_latents import CONFIG

def main(output_directory=CONFIG["output_path"]/"bubblewrap_runs", steps_to_run=None):
    obs, beh = simulate_example_data()
    in_ds = NumpyTimedDataSource(obs, None, time_offsets=(1,))
    out_ds = NumpyTimedDataSource(beh, None, time_offsets=(1,))

    # define the adaptive_latents object
    bw = Bubblewrap(dim=in_ds.output_shape,  **default_rwd_parameters)

    # define the (optional) method to regress the HMM state from `bw.alpha`
    reg = SemiRegularizedRegressor(input_d=bw.N, output_d=1)

    class CustomAnimation(AnimationManager):
        n_rows = 1
        n_cols = 1
        figsize = (15,10)
        extension = "mp4"

        def custom_draw_frame(self, step, bw: Bubblewrap, br: BWRun):
            historical_observations, _ = br.input_ds.get_history()

            pfs.show_bubbles_2d(self.ax[0,0], historical_observations, bw, alpha_coefficient=.5)
            self.ax[0,0].set_title(f"Step {step}")
        def frame_draw_condition(self, step_number, bw):
            return step_number % 5 == 0
    am = CustomAnimation()

    # define the object to coordinate all the other objects
    br = BWRun(bw=bw, in_ds=in_ds, out_ds=out_ds, behavior_regressor=reg, animation_manager=am, show_tqdm=True, output_directory=output_directory)

    br.run(limit=steps_to_run, save=True)

if __name__ == '__main__':
    main(steps_to_run=100)
