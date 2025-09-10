from adaptive_latents import StreamingKalmanFilter, StimRegressor, BaseMultiKernelRegressor
from adaptive_latents.input_sources.lds_simulation import LDS

"""
Demo: Stimulus-response regression
"""

def main(show_plots=True):
    _, observations, stimulations = LDS.run_nest_dynamical_system(rotations=10, u_function='curvy')

    sr = StimRegressor(
        autoreg=StreamingKalmanFilter(steps_between_refits=5),
        stim_reg=BaseMultiKernelRegressor(maxlen=10),
    )

    sr.offline_run_on([(observations, 'X'),  (stimulations, 'stim')])


if __name__ == '__main__':
    main()