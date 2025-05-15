import jax
import time

from adaptive_latents import StreamingKalmanFilter, ArrayWithTime, Pipeline, StimRegressor, Bubblewrap, proSVD, CenteringTransformer, VJF, KernelSmoother
from adaptive_latents.regressions import BaseKernelRegressor
import numpy as np
from adaptive_latents.stim_designer import StimDesigner

def make_times(input_array, rng):
    decay_rate = .8
    stim_magnitude = 10
    max_l0_norm = 30
    n_identity_prior = 10
    initial_nostim_period = 5
    stim_reg_maxlen = 500
    centerer_init_size = 0


    smoothing_tau = .16 # TODO: I think this isn't in the one figure?
    stim_rate = 1 / 2
    prosvd_k = 10
    true_S = 'identity'
    stim_direction_type = 'first'



    stim_time_rng, other_rng = rng.spawn(2)
    sr = StimRegressor(
        autoreg=StreamingKalmanFilter(steps_between_refits=50), # TODO: optimize
        stim_reg=BaseKernelRegressor(length_scale=0.04, maxlen=stim_reg_maxlen),
        stim_designer=StimDesigner(max_l0_norm=max_l0_norm, rng_seed=other_rng.integers(2 ** 32), should_log=False),
        log_level=0,
        check_dt=True,
        attempt_correction=True,
        heed_stimuli=True,
        stim_delay=0,
    )

    centerer = CenteringTransformer(init_size=centerer_init_size, nan_when_uninitialized=True)
    smoother = KernelSmoother(tau=smoothing_tau/input_array.dt)
    pro = proSVD(k=prosvd_k)


    times = []
    to_add = np.zeros(input_array.shape[1])
    for data in Pipeline().streaming_run_on(input_array):
        times.append(time.time())

        stim_decision = data.t > initial_nostim_period and stim_time_rng.random() < stim_rate * input_array.dt
        if stim_decision and pro.Q is not None:
            if stim_direction_type == 'first':
                desired_stim = np.zeros((pro.Q.shape[1], 1))
                desired_stim[0] = 1
            elif stim_direction_type == 'col':
                desired_stim = np.zeros((pro.Q.shape[1], 1))
                desired_stim[other_rng.choice(pro.Q.shape[1]), 0] = 1
            elif stim_direction_type == 'random':
                desired_stim = other_rng.normal(size=(pro.Q.shape[1], 1))
                desired_stim = desired_stim / np.linalg.norm(desired_stim)
            else:
                raise ValueError()


            if sr.stim_reg.n_observed > n_identity_prior:
                f = sr.stim_reg.make_jax_pred_f()
                def u_to_s_function(u):
                    return stim_magnitude * f(jax.numpy.hstack((sr.autoreg.predict(n_steps=0), u)))
            else:
                def u_to_s_function(u):
                    return stim_magnitude * pro.Q.T @ u

            designed_stim, _ = sr.stim_designer.design_stim(desired_stim, u_to_s_function=u_to_s_function, u_dimension=pro.Q.shape[0])
            instantaneous_stim = designed_stim * stim_magnitude
        else:
            instantaneous_stim = np.zeros(input_array.shape[1])



        if true_S == 'identity':
            transformed_instantaneous_stim = instantaneous_stim
        elif true_S == 'flip':
            if pro.Q is not None:
                in_space_comp = pro.Q.T @ instantaneous_stim
                out_of_space_comp = instantaneous_stim - pro.Q @ in_space_comp
                transformed_instantaneous_stim = pro.Q @ in_space_comp[::-1] + out_of_space_comp
            else:
                assert (instantaneous_stim == 0).all()
                transformed_instantaneous_stim = instantaneous_stim
        else:
            raise ValueError(true_S)

        to_add = to_add + transformed_instantaneous_stim
        data = data + to_add
        to_add = decay_rate * to_add
        data = centerer.partial_fit_transform(data, stream= 'X')
        data = smoother.partial_fit_transform(data, stream= 'X')
        data = pro.partial_fit_transform(data, stream='X')


        sr.partial_fit_transform(ArrayWithTime(instantaneous_stim, data.t), stream= 'stim')
        data = sr.partial_fit_transform(data, stream= 'X')

    return times



if __name__ == '__main__':
    from adaptive_latents import datasets
    d = datasets.Odoherty21Dataset()
    rng = np.random.default_rng(0)
    times = make_times(input_array=d.neural_data, rng=rng)
    np.save('times.npy', times)

    import matplotlib.pyplot as plt
    plt.plot(np.diff(times))
    plt.show()