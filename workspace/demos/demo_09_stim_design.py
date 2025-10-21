from adaptive_latents import StreamingKalmanFilter, StimRegressor, BaseMultiKernelRegressor, ArrayWithTime
from adaptive_latents.stim_designer import StimDesigner
from adaptive_latents.input_sources.lds_simulation import LDS
import numpy as np

"""
Demo: Stimulus design
"""

def make_stims_3d(stimulations, rng): # TODO: move this to lds_simulation
    def angle_beween(v1, v2):
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return angle

    new_stimulations = []
    for i, stimulation in enumerate(stimulations):
        new_stimulation = np.zeros(3)
        if stimulation.any():
            new_stimulation[2] = stimulation[0]
        elif i > 0 and stimulations[i - 1].any():
            a_between = 0
            while a_between < 5 * np.pi / 180:
                new_stimulation = rng.normal(size=(3,))
                new_stimulation = new_stimulation / np.linalg.norm(new_stimulation)
                new_stimulation = np.abs(new_stimulation)
                a_between = angle_beween(new_stimulation, np.array([0, 0, 1]))
        new_stimulations.append(ArrayWithTime(new_stimulation, stimulation.t))
    return ArrayWithTime.from_list(new_stimulations)

def main(show_plots=True):
    _, observations, stimulations = LDS.run_nest_dynamical_system(rotations=100, u_function='curvy')

    stimulations = make_stims_3d(stimulations, np.random.default_rng(0))

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
    regressed_u_to_s_function = lambda u: sr.stim_reg.make_jax_pred_f()([np.array([1, 0, 0]), u, observations.t[-1]])
    u = stim_designer_closed_loop.design_stim(
        goal,
        u_dimension=observations.shape[1],
        u_to_s_function= regressed_u_to_s_function,
    )
    print(u)


if __name__ == '__main__':
    main()