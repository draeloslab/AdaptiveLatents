# import os
# os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from adaptive_latents.sim_stim import make_sr
from adaptive_latents.stim_designer import StimDesigner
import numpy as np

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
    #     **kwargs,
    # )
    # stim_designer_1: StimDesigner

    # print(stim_designer_1.log[0].keys()) # see StimDesigner.design_stim and StimDesigner.design_stim_jaxopt for what is logged; you should see variables you recognize like 'v' and 'u'

    #run closed loop
    stim_regressor, stim_designer_2, log = make_sr(
        rng=np.random.default_rng(0),
        u_to_s_model_type='kernel_regressed',
        log_dir=SESSION_FOLDER,
        **kwargs,
    )

    f_pred = stim_regressor.stim_reg.make_jax_pred_f()
    import jax
    import jax.numpy as jnp
    # pick some reasonable state/time; doesn’t have to be perfect
    pred = stim_regressor.autoreg.predict(n_steps=0)
    u_test = np.zeros(stim_regressor.stim_reg.input_histories[1].shape[-1])  # or just np.zeros(num_neurons)
    t_test = np.array([float(log['latents'].t[-1])])

    s_test = f_pred([jnp.asarray(pred), jnp.asarray(u_test), jnp.asarray(t_test)])
    print("DEBUG kernel output finite?", np.all(np.isfinite(np.array(s_test))))
    print("DEBUG kernel output:", np.array(s_test)[:10])

    #print(stim_designer_2.log[0]['equiv_proj_mat'] - stim_designer_1.log[0]['equiv_proj_mat']) # entries of the two logs correspond to the same point in simulation time