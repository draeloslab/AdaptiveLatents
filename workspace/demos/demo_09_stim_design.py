from adaptive_latents import StreamingKalmanFilter, StimRegressor, BaseMultiKernelRegressor
from adaptive_latents.stim_designer import StimDesigner
from adaptive_latents.input_sources.lds_simulation import LDS
import numpy as np

"""
Demo: Stimulus design
"""

def main(show_plots=True):
    _, observations, stimulations = LDS.run_nest_dynamical_system(rotations=100, u_function='curvy')

    sr = StimRegressor(
        autoreg=StreamingKalmanFilter(steps_between_refits=5),
        stim_reg=BaseMultiKernelRegressor(maxlen=10),
    )

    sr.offline_run_on([(observations, 'X'),  (stimulations, 'stim')])


    goal = np.zeros((3,1))
    goal[2] = 1

    stim_designer_open_loop = StimDesigner(optimization_method='jaxopt', should_log=True)
    u = stim_designer_open_loop.design_stim(
        goal,
        u_dimension=3,
        u_to_s_function=lambda u: u,
    )
    print(u)
    # check out `stim_designer_open_loop.log`


    stim_designer_closed_loop = StimDesigner(optimization_method='jaxopt')
    regressed_u_to_s_function = sr.stim_reg.make_jax_pred_f()
    u = stim_designer_closed_loop.design_stim(
        goal,
        u_dimension=observations.shape[1],
        u_to_s_function= regressed_u_to_s_function,
    )
    print(u)


if __name__ == '__main__':
    main()