import warnings

import numpy as np
from scipy.stats import special_ortho_group

from adaptive_latents.timed_data_source import ArrayWithTime

from .kalman_filter import KalmanFilter


class LDS:
    def __init__(self, A, C, W, Q, B=None, state_center=None, observation_center=None):
        self.A = A
        self.C = C
        self.W = W
        self.Q = Q

        if (self.W == 0).all():
            self.W_cholesky = 0 * self.W
        else:
            self.W_cholesky = np.linalg.cholesky(self.W)

        if (self.Q == 0).all():
            self.Q_cholesky = 0 * self.Q
        else:
            self.Q_cholesky = np.linalg.cholesky(self.Q)

        self.B = B if B is not None else np.zeros((0, A.shape[0]))
        self.state_center = state_center if state_center is not None else 0
        self.observation_center = observation_center if observation_center is not None else 0
        self.check_shapes_correct()

    def check_shapes_correct(self):
        assert self.A is not None
        assert self.A.shape == self.W.shape
        assert self.A.shape[1] == self.C.shape[0] == self.B.shape[1]
        assert self.C.shape[1] == self.Q.shape[1] == self.Q.shape[0]
        assert np.allclose(self.Q, self.Q.T)
        assert np.allclose(self.W, self.W.T)

    def simulate(self, n_steps, initial_state=None, U=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        if isinstance(U, np.ndarray):
            assert U.shape[0] == n_steps
            def u_function(lds, state, i, rng):
                return U[i]
        else:
            u_function = U

        states = np.zeros((n_steps, self.A.shape[0]))
        observations = np.zeros((n_steps, self.C.shape[1]))
        control = np.zeros((n_steps, self.B.shape[0]))

        if initial_state is not None:
            states[0] = np.array(initial_state) - self.state_center
        else:
            states[0,:] = 0
            warnings.warn("simulating with the initial state in equilibrium")

        state = states[0]
        for i in range(n_steps):
            state, observation, u = self.simulate_step(state, rng, u_function, i, use_state_dynamics=i != 0, add_centers=False)
            states[i] = state
            observations[i] = observation
            control[i] = u

        return states + self.state_center, observations + self.observation_center, control
        # return ArrayWithTime.from_notime(states + self.state_center), ArrayWithTime.from_notime(observations + self.observation_center), ArrayWithTime.from_notime(control)

    def simulate_step(self, state, rng, u_function=None, i=None, use_state_dynamics=True, add_centers=True):
        if add_centers:
            state = state - self.state_center

        u = np.array([])
        if u_function is not None and isinstance(u_function, np.ndarray):
            u = u_function
        elif u_function is not None:
            u = u_function(lds=self, state=state, i=i, rng=rng)

        if use_state_dynamics:  # I don't want this sometimes on the first iteration
            state = state @ self.A
            random_jitter = rng.normal(size=self.A.shape[1]) @ self.W_cholesky
            state = state + random_jitter

        state += u @ self.B

        observation = state @ self.C
        observation_noise = rng.normal(size=self.C.shape[1]) @ self.Q_cholesky
        observation = observation + observation_noise

        if add_centers:
            state = state + self.state_center
            observation = observation + self.observation_center

        return state, observation, u


    @classmethod
    def from_kalman_filter(cls, kf):
        kf: KalmanFilter
        return cls(kf.A, kf.C, kf.W, kf.Q, state_center=kf.X_mean, observation_center=kf.Y_mean)

    @classmethod
    def circular_lds(cls, transitions_per_rotation=30 + 1 / np.pi, obs_d=10, process_noise=0.01, obs_noise=0.02, obs_center=0, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        theta = 2*np.pi/transitions_per_rotation
        A = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        C = special_ortho_group(dim=obs_d, seed=rng).rvs()[:, :2].T
        lds = cls(A, C, np.eye(2) * process_noise, np.eye(obs_d) * obs_noise, state_center=0, observation_center=obs_center)
        lds.transitions_per_rotation = transitions_per_rotation
        return lds

    @classmethod
    def nest_lds(cls, transitions_per_rotation=30 + 1 / np.pi, rng=None, noise=0.05):
        rng = rng if rng is not None else np.random.default_rng()
        base_lds = LDS.circular_lds(transitions_per_rotation=transitions_per_rotation, rng=rng)

        A = base_lds.A
        A = np.hstack([A, np.zeros((A.shape[0], 1))])
        A = np.vstack([A, np.zeros((1, A.shape[1]))])
        A[-1, -1] = .8
        C = np.eye(A.shape[1])
        B = np.eye(A.shape[1])
        W = np.eye(A.shape[1]) * noise
        Q = np.eye(A.shape[1]) * noise
        return LDS(A, C, W, Q, B=B)

    @classmethod
    def run_nest_dynamical_system(cls, rotations, transitions_per_rotation=30 + 1 / np.pi, stim_magnitude=1, stims_per_rotation=1, radius=5, u_function=None, rng=None, early_shift=1e-12, noise=0.05):
        # rng = rng if rng is not None else np.random.default_rng()
        # dynamics_rng, stim_rng = rng.spawn(2)

        if rng is None:
            dynamics_rng = np.random.default_rng()
            stim_rng = np.random.default_rng()
        else: 
            base_ss = np.random.SeedSequence(int(rng))   
            ss_dyn, ss_stim = base_ss.spawn(2)
            dynamics_rng = np.random.default_rng(ss_dyn)
            stim_rng = np.random.default_rng(ss_stim)


        lds = cls.nest_lds(transitions_per_rotation=transitions_per_rotation, rng=dynamics_rng, noise=noise)
        N = int(rotations * transitions_per_rotation)
        t = np.linspace(0, N / transitions_per_rotation, N)

        stim = t * 0
        stim[stim_rng.choice(stim.shape[0], size=int(stims_per_rotation * N / transitions_per_rotation), replace=False)] = 1

        if u_function == 'curvy':
            def u_function(lds, state, i, rng):
                u = np.zeros(lds.B.shape[0])
                u[2] = stim_magnitude * stim[i] * state[0] / np.linalg.norm(state[:2])
                return u
        elif u_function == 'curvy flips':
            def u_function(lds, state, i, rng):
                u = np.zeros(lds.B.shape[0])
                u[2] = stim_magnitude * stim[i] * state[0] / np.linalg.norm(state[:2]) * (-1 if i > stim.shape[0]//2 else 1)
                return u

        elif u_function == 'constant':
            def u_function(lds, state, i, rng):
                u = np.zeros(lds.B.shape[0])
                u[2] = stim_magnitude * stim[i]
                return u
        elif u_function is None:
            u_function = lambda **_: np.zeros(lds.B.shape[0])

        states, observations, received_stim = lds.simulate(N, initial_state=[radius, 0, 0], U=u_function, rng=dynamics_rng)

        assert early_shift == 0 or np.diff(t).mean() / early_shift > 100

        stim = ArrayWithTime(stim[:,None], t - 2*early_shift)
        X = ArrayWithTime(states, t - 1*early_shift)
        Y = ArrayWithTime(observations, t - 0*early_shift)

        return X, Y, stim
