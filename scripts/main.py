import numpy as np
from bubblewrap import Bubblewrap, BWRun, AnimationManager
from bubblewrap.regressions import NearestNeighborRegressor, SymmetricNoisyRegressor
from bubblewrap.default_parameters import default_jpca_dataset_parameters
from bubblewrap.input_sources.functional import get_from_saved_npz
from bubblewrap.input_sources.data_sources import NumpyPairedDataSource
import bubblewrap.plotting_functions as bpf

def example_movie(shorten):
    obs, beh = get_from_saved_npz("jpca_reduced_sc.npz")
    ds = NumpyPairedDataSource(obs, beh, time_offsets=(1,))

    ds.shorten(shorten)

    # define the bubblewrap object
    bw = Bubblewrap(dim=ds.output_shape[0], copy_row_on_teleport=False, **dict(default_jpca_dataset_parameters, B_thresh=-15, seed=shorten))

    # define the (optional) method to regress the HMM state from `bw.alpha`
    # reg = SymmetricNoisyRegressor(input_d=bw.N, output_d=1)
    reg = NearestNeighborRegressor(input_d=bw.N, output_d=1, maxlen=600)

    class CustomAnimation(AnimationManager):
        n_rows = 2
        n_cols = 3
        figsize = (15,10)
        extension = "mp4"

        def setup(self):
            gs = self.ax[0,0].get_gridspec()
            for ax in self.ax[:,0]:
                ax.remove()
            self.tall_ax = self.fig.add_subplot(gs[:,0])

        def custom_draw_frame(self, step, bw: Bubblewrap, br: BWRun):
            historical_observations = br.data_source.get_history()[0]

            bpf.show_bubbles_2d(self.tall_ax, historical_observations, bw, alpha_coefficient=.5)
            self.tall_ax.set_title(f"Step {step}")
            bpf.show_alphas_given_regression_value(self.ax[0,1], br, behavior_value=10, step=step, history_length=50, hist_axis=self.ax[1,1])
            bpf.show_alphas_given_regression_value(self.ax[0,2], br, behavior_value=-10, step=step, history_length=50, hist_axis=self.ax[1,2])

        def frame_draw_condition(self, step_number, bw):
            return step_number % 5 == 0
    am = CustomAnimation()

    # define the object to coordinate all the other objects
    br = BWRun(bw=bw, data_source=ds, behavior_regressor=reg, animation_manager=am, show_tqdm=True)

    # run and save the output
    br.run()

def main():
    example_movie(0)


if __name__ == '__main__':
    main()
