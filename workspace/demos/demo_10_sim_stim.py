from adaptive_latents.sim_stim import make_sr
from adaptive_latents.stim_designer import StimDesigner
import numpy as np

import jax
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)


if __name__ == '__main__':
    from adaptive_latents import datasets
    d = datasets.Odoherty21Dataset()
    rng = np.random.default_rng(42)

    SESSION_FOLDER = r"C:\Users\secom\OneDrive\Documents\DraelosLab\AdaptiveLatents\AdaptiveLatents\workspace\Alexworkspace\metricsfolder"

    kwargs = dict(
        input_array=d.neural_data,
        show_tqdm=True,
        exit_time=60, # this lets you stop before the end of the recording; 60 means to stop after 60 seconds, np.inf means run until the end
        stim_rate=2, # a stim every 2 seconds
        stim_timing_method='isi', # makes sure the stims are regular instead of random
    )

    # run open loop
    # stim_regressor, stim_designer_1, log = make_sr(
    #     rng=np.random.default_rng(0),
    #     u_to_s_model_type='identity',
    #     log_dir=SESSION_FOLDER,
    #     optimization_method = 'admm',
    #     **kwargs,
    # )
    # stim_designer_1: StimDesigner

    # print(stim_designer_1.log[0].keys()) # see StimDesigner.design_stim and StimDesigner.design_stim_jaxopt for what is logged; you should see variables you recognize like 'v' and 'u'

    #run closed loop
    stim_regressor, stim_designer_2, log = make_sr(
        rng=np.random.default_rng(0),
        u_to_s_model_type='kernel_regressed',
        log_dir=SESSION_FOLDER,
        optimization_method = 'admm',
        **kwargs,
    )
    #print(stim_designer_2.log[0]['equiv_proj_mat'] - stim_designer_1.log[0]['equiv_proj_mat']) # entries of the two logs correspond to the same point in simulation time