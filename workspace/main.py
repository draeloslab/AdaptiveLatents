from adaptive_latents import Bubblewrap, BWRun, AnimationManager, SemiRegularizedRegressor, CONFIG
from adaptive_latents.input_sources.hmm_simulation import simulate_example_data
import adaptive_latents.plotting_functions as pfs

def main(output_directory=CONFIG['bwrun_save_path'], steps_to_run=None, make_animation=True):
    # set make_animation to `False` to run much faster
    obs, beh = simulate_example_data(dimensionality=2)

    # define the adaptive_latents object
    bw = Bubblewrap(dim=obs.shape[1])

    # define the (optional) method to regress the HMM state from `bw.alpha`
    reg = SemiRegularizedRegressor(input_d=bw.N, output_d=1)

    class CustomAnimation(AnimationManager):
        n_rows = 1
        n_cols = 1
        fps = 10
        figsize = (15,10)
        extension = "mp4"

        def custom_draw_frame(self, step, bw: Bubblewrap, br: BWRun):
            historical_observations, _ = br.input_ds.get_history()

            pfs.show_bubbles_2d(self.ax[0,0], historical_observations, bw, alpha_coefficient=.5)
            self.ax[0,0].set_title(f"Step {step}")

        def frame_draw_condition(self, step_number, bw):
            return step_number % 5 == 0

    am = CustomAnimation() if make_animation else False

    # define the object to coordinate all the other objects
    br = BWRun(bw=bw, in_ds=obs, out_ds=beh, behavior_regressor=reg, animation_manager=am, show_tqdm=True, output_directory=output_directory)

    br.run(bw_step_limit=steps_to_run, save_bw_history=False)

if __name__ == '__main__':
    main(output_directory='.',steps_to_run=500)
