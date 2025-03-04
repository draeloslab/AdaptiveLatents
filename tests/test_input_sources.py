import adaptive_latents.input_sources as ins
import numpy as np
import pytest


def test_hmm_runs(rng):
    # note I do not test correctness here
    for hmm in (
        ins.hmm_simulation.HMM.gaussian_clock_hmm(n_states=10),
        ins.hmm_simulation.HMM.gaussian_clock_hmm(n_states=10, high_d_pad=10),
        ins.hmm_simulation.HMM.wandering_gaussian_clock_hmm(n_states=10),
        ins.hmm_simulation.HMM.teetering_gaussian_clock_hmm(n_states=10),
        ins.hmm_simulation.HMM.inverting_gaussian_clock_hmm(n_states=10),
        ins.hmm_simulation.HMM.discrete_clock_hmm(n_states=10),
        ins.hmm_simulation.HMM.wave_clock_hmm(n_states=11),
        ins.hmm_simulation.HMM.infinity_shape_hmm(n_states=11),
    ):
        hmm.simulate(50, rng)
        states, observations = hmm.simulate_with_states(10, rng)
        hmm.advance_one_step(rng, states[-1])


class TestAutoRegressor:
    @pytest.mark.parametrize('rank_limit', [2, None])
    def test_can_regress(self, rng, rank_limit):
        plot = False
        if plot:
            import matplotlib.pyplot as plt

        samples_per_second = 20
        seconds_per_rotation = 3
        total_seconds = 5

        t = np.linspace(0, total_seconds/seconds_per_rotation * np.pi * 2, total_seconds*samples_per_second+1)
        X = np.vstack((np.cos(t), np.sin(t))).T + 10
        stim = np.zeros(shape=(X.shape[0],1))

        ar = ins.autoregressor.AR_K(k=1, rank_limit=rank_limit, init_method='full_rank', iter_limit=500, rng=rng)
        ar.fit(X, stim)

        if plot:
            plt.plot(X[:,0], X[:,1])

        initial_point = np.array([[10,12]])
        X_hat = ar.predict(initial_observations=initial_point, n_steps=samples_per_second*seconds_per_rotation)

        one_step_distance = np.linalg.norm(X[0]-X[1])
        def close(a, b, radius):
            return np.linalg.norm(a-b) < radius

        if plot:
            plt.plot([initial_point[0,0], X_hat[0,0]], [initial_point[0,1], X_hat[0,1]], '--.', color='C1')
            plt.plot(X_hat[:,0], X_hat[:,1], '.-')

        assert close(X_hat[-1], initial_point, 2*one_step_distance)
        assert not close(X_hat[X_hat.shape[0]//2], initial_point, 2*one_step_distance)

        initial_point = np.array([[10,10]])
        X_hat = ar.predict(initial_observations=initial_point, n_steps=samples_per_second*seconds_per_rotation)
        if plot:
            plt.plot([initial_point[0,0], X_hat[0,0]], [initial_point[0,1], X_hat[0,1]], '--.', color='C2')
            plt.plot(X_hat[:,0], X_hat[:,1], '.-')
        assert close(X_hat[-1], initial_point, .1*one_step_distance)
        assert close(X_hat[X_hat.shape[0]//2], initial_point, .3)
        if plot:
            plt.axis('equal')
