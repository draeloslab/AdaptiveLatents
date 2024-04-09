import os
import sys

match sys.argv[1]:
    case "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    case "gpu":
        pass
    case _:
        raise Exception()

from adaptive_latents import Bubblewrap, BWRun, AnimationManager, NumpyTimedDataSource, default_rwd_parameters, CONFIG
import adaptive_latents.input_sources.utils as fin
import adaptive_latents
import numpy as np

def main():
    # t = np.arange(0, 2*np.pi*50, np.pi/5)
    # X = np.column_stack([np.cos(t), np.sin(t)])
    # X = X + np.random.default_rng(0).normal(size=X.shape)*1e-3
    # in_ds = NumpyTimedDataSource(X, t, time_offsets=(0,1))

    obs, raw_behavior, bin_centers, beh_t = adaptive_latents.input_sources.datasets.construct_buzaki_data(individual_identifier=adaptive_latents.input_sources.datasets.individual_identifiers["buzaki"][0], bin_width=0.03)
    ###
    resampled_behavior = fin.resample_matched_timeseries(raw_behavior, bin_centers, beh_t)
    hd = np.arctan2(resampled_behavior[:,0] - resampled_behavior[:,2], resampled_behavior[:,1] - resampled_behavior[:,3])
    beh = resampled_behavior[:,:2]

    # X, t = fin.clip(fin.prosvd_data(input_arr=fin.zscore(np.hstack([obs, beh])), output_d=6, init_size=50), bin_centers)
    X, t = fin.clip(fin.prosvd_data(input_arr=(np.hstack([obs, beh])), output_d=6, init_size=50), bin_centers)

    in_ds = NumpyTimedDataSource(X, t, time_offsets=(0,1))

    bw_params = dict(    
    adaptive_latents.default_parameters.default_jpca_dataset_parameters, 
    num=100,
    eps=1e-3,
    step=1,
    num_grad_q=3,
    )

    bw = Bubblewrap(dim=in_ds.output_shape, **bw_params)


    # class CustomAnimation(AnimationManager):
    #     n_rows = 1
    #     n_cols = 1
    #     figsize = (15,10)
    #     extension = "mp4"
    #
    #     def custom_draw_frame(self, step, bw: Bubblewrap, br: BWRun):
    #         historical_observations, _ = br.input_ds.get_history()
    #
    #         adaptive_latents.plotting_functions.show_bubbles_2d(self.ax[0,0], historical_observations, bw, alpha_coefficient=.5)
    #         self.ax[0,0].set_title(f"Step {step}")
    #     def frame_draw_condition(self, step_number, bw):
    #         return step_number % 5 == 0
    # am = CustomAnimation()


    # define the object to coordinate all the other objects
    br = BWRun(bw=bw, in_ds=in_ds, show_tqdm=True, log_level=100)

    br.run(save=True, limit=63)
    # print(f"{numpy.array(bw.L)[0][0][0]:.32f} ({ xla_bridge.get_backend().platform })")

if __name__ == '__main__':
    main()




