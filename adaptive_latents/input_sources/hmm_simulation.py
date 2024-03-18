import numpy as np


class GaussianEmissionModel:
    def __init__(self, means, covariances):
        self.means = means
        self.covariances = covariances
        self.embedded_dimension = means.shape[1]
        self.number_of_states = means.shape[0]

    def get_observation(self, bubble, rng):
        return rng.multivariate_normal(self.means[bubble, ...], self.covariances[bubble, ...])


def make_p(X):
    "A utility function to convert numpy matrices into probability matrices."
    return X / X.sum(axis=1)[:, None]


class DiscreteEmissionModel:
    def __init__(self, observation_matrix):
        self.observation_matrix = observation_matrix
        self.number_of_states = observation_matrix.shape[0]
        self.number_of_output_characters = observation_matrix.shape[1]
        self.embedded_dimension = 1

    def get_observation(self, state, rng):
        pvec = self.observation_matrix[state, :]
        return rng.choice(self.number_of_output_characters, p=pvec)


class HMM:
    def __init__(self, transition_matrix, emission_model, initial_distribution, mutation_function=None):
        """
        mutation_function should have the signature (HMM, time, [states]) -> (transition_matrix, emission_model, initial_distribution)
        """
        self.n_states = transition_matrix.shape[0]
        self.n_symbols = emission_model.number_of_states
        self.transition_matrix = transition_matrix
        self.emission_model = emission_model
        self.initial_distribution = initial_distribution
        self.mutation_function = mutation_function
        self.sanity_check()

    def sanity_check(self):
        assert np.allclose(self.transition_matrix.sum(axis=1), 1)
        assert np.all(self.transition_matrix >= 0)

    @staticmethod
    def gaussian_clock_hmm(n_states=12, p1=.5, angle=0., variance_scale=1., radius=10, high_d_pad=0):
        transition_matrix = np.eye(n_states)
        diag_indices = list(np.diag_indices(n_states))
        diag_indices[1] = np.roll(diag_indices[1], -1)
        transition_matrix[tuple(diag_indices)] = p1

        transition_matrix[np.diag_indices(n_states)] = 1 - p1

        initial_distribution = np.zeros(n_states)
        initial_distribution[0] = 1

        data_dimension = 2 + high_d_pad
        means = np.zeros((n_states, data_dimension))
        variances = np.stack([np.eye(data_dimension) * variance_scale for _ in range(n_states)])

        for i in range(n_states):
            theta = 2 * np.pi * i / n_states + angle
            means[i, :] = np.array([np.cos(theta), np.sin(theta)] + [0] * high_d_pad) * radius

        return HMM(transition_matrix, GaussianEmissionModel(means, variances), initial_distribution)

    @staticmethod
    def wandering_gaussian_clock_hmm(n_states, p1=1., angle=0., radius=10, speed=0):
        transition_matrix = np.eye(n_states)
        diag_indices = list(np.diag_indices(n_states))
        diag_indices[1] = np.roll(diag_indices[1], -1)
        transition_matrix[tuple(diag_indices)] = p1

        transition_matrix[np.diag_indices(n_states)] = 1 - p1

        initial_distribution = np.zeros(n_states)
        initial_distribution[0] = 1

        data_dimension = 2
        means = np.zeros((n_states, data_dimension))
        variances = np.stack([np.eye(data_dimension) for _ in range(n_states)])

        for i in range(n_states):
            theta = 2 * np.pi * i / n_states + angle
            means[i, :] = np.array([np.cos(theta), np.sin(theta)]) * radius

        def mutation_function(hmm, time, rng):
            means = hmm.emission_model.means
            means = means + rng.normal(size=means.shape) * speed
            em = GaussianEmissionModel(means=means, covariances=hmm.emission_model.covariances)

            return (hmm.transition_matrix, em, hmm.initial_distribution)

        return HMM(transition_matrix, GaussianEmissionModel(means, variances), initial_distribution,
                   mutation_function=mutation_function)

    @staticmethod
    def teetering_gaussian_clock_hmm(n_states, p0=0, p1=1., angle=0., rate=1., radius=10):
        """p1 is the probability of switching"""

        transition_matrix = np.eye(n_states)
        diag_indices = list(np.diag_indices(n_states))
        diag_indices[1] = np.roll(diag_indices[1], -1)
        transition_matrix[tuple(diag_indices)] = p1

        transition_matrix[np.diag_indices(n_states)] = 1 - p1

        initial_distribution = np.zeros(n_states)
        initial_distribution[0] = 1

        data_dimension = 2
        means = np.zeros((n_states, data_dimension))
        variances = np.stack([np.eye(data_dimension) for _ in range(n_states)])

        for i in range(n_states):
            theta = 2 * np.pi * i / n_states + angle
            means[i, :] = np.array([np.cos(theta), np.sin(theta)]) * radius

        def mutation_function(hmm, time, rng):
            m_transition_matrix = np.eye(n_states)
            diag_indices = list(np.diag_indices(n_states))
            diag_indices[1] = np.roll(diag_indices[1], -1)

            mixing_v = (np.sin(time * rate) / 2 + .5)
            switch_p = mixing_v * p1 + (1 - mixing_v) * p0
            m_transition_matrix[tuple(diag_indices)] = switch_p

            m_transition_matrix[np.diag_indices(n_states)] = 1 - switch_p

            return (m_transition_matrix, hmm.emission_model, hmm.initial_distribution)

        return HMM(transition_matrix, GaussianEmissionModel(means, variances), initial_distribution,
                   mutation_function=mutation_function)

    @staticmethod
    def inverting_gaussian_clock_hmm(n_states, mixing_p=1, p1=1, angle=0., rate=1, radius=10):
        "p1 is the probability of switching"
        transition_matrix = np.eye(n_states)
        transition_matrix[np.diag_indices(n_states)] = 1

        initial_distribution = np.zeros(n_states)
        initial_distribution[0] = 1

        data_dimension = 2
        means = np.zeros((n_states, data_dimension))
        variances = np.stack([np.eye(data_dimension) for _ in range(n_states)])

        for i in range(n_states):
            theta = 2 * np.pi * i / n_states + angle
            means[i, :] = np.array([np.cos(theta), np.sin(theta)]) * radius

        def mutation_function(hmm, time, rng):
            m_transition_matrix = np.eye(n_states)

            forward_diag_indices = list(np.diag_indices(n_states))
            forward_diag_indices[1] = np.roll(forward_diag_indices[1], -1)

            backward_diag_indices = list(np.diag_indices(n_states))
            backward_diag_indices[1] = np.roll(backward_diag_indices[1], 1)

            s = np.sin(time * rate) * p1
            v = np.array([s - -1, abs(s), 1 - s])
            v = -v
            v = np.exp(v * mixing_p)
            v = v / v.sum()

            m_transition_matrix[tuple(forward_diag_indices)] = v[2]
            m_transition_matrix[tuple(backward_diag_indices)] = v[0]
            m_transition_matrix[np.diag_indices(n_states)] = v[1]

            return (m_transition_matrix, hmm.emission_model, hmm.initial_distribution)

        return HMM(transition_matrix, GaussianEmissionModel(means, variances), initial_distribution,
                   mutation_function=mutation_function)

    @staticmethod
    def discrete_clock_hmm(n_states, p1=1.0):
        transition_matrix = np.eye(n_states)
        diag_indices = list(np.diag_indices(n_states))
        diag_indices[1] = np.roll(diag_indices[1], -1)
        transition_matrix[tuple(diag_indices)] = p1
        transition_matrix = make_p(transition_matrix)

        initial_distribution = np.zeros(n_states)
        initial_distribution[0] = 1

        observation_matrix = make_p(np.eye(n_states) + .2)

        return HMM(transition_matrix, DiscreteEmissionModel(observation_matrix), initial_distribution)

    @staticmethod
    def wave_clock_hmm(n_states=21, displacement_center=3, displacement_spread=.3, variance_scale=1., radius=10, angle=0):
        assert n_states % 2 == 1  # this just makes things easier with the center of the hill

        p = np.exp(-1/2 * (np.linspace(-1,1,n_states)/displacement_spread)**2)
        p /= p.sum()
        p = np.roll(p, shift=(p.shape[0] // 2 + 1))

        transition_matrix = np.array([np.roll(p, shift=shift + displacement_center) for shift in range(n_states)])



        initial_distribution = np.zeros(n_states)
        initial_distribution[0] = 1

        data_dimension = 2
        means = np.zeros((n_states, data_dimension))
        variances = np.stack([np.eye(data_dimension) * variance_scale for _ in range(n_states)])

        for i in range(n_states):
            theta = 2 * np.pi * i / n_states + angle
            means[i, :] = np.array([np.cos(theta), np.sin(theta)]) * radius

        return HMM(transition_matrix, GaussianEmissionModel(means, variances), initial_distribution)

    @staticmethod
    def infinity_pool_hmm(n_states=21, displacement_center=3, displacement_spread=.3, variance_scale=1., a=10, angle=0):
        assert n_states % 2 == 1  # this just makes things easier with the center of the hill

        p = np.exp(-1/2 * (np.linspace(-1,1,n_states)/displacement_spread)**2)
        p /= p.sum()
        p = np.roll(p, shift=(p.shape[0] // 2 + 1))

        transition_matrix = np.array([np.roll(p, shift=shift + displacement_center) for shift in range(n_states)])



        initial_distribution = np.zeros(n_states)
        initial_distribution[0] = 1

        data_dimension = 2
        means = np.zeros((n_states, data_dimension))
        variances = np.stack([np.eye(data_dimension) * variance_scale for _ in range(n_states)])

        for i in range(n_states):
            theta = 2 * np.pi * i / n_states
            x_coord = a * np.cos(theta) / (1 + np.sin(theta)**2)
            y_coord = a * np.cos(theta) * np.sin(theta)/ (1 + np.sin(theta)**2)
            means[i, :] = np.array([x_coord, y_coord])

        return HMM(transition_matrix, GaussianEmissionModel(means, variances), initial_distribution)

    def simulate_with_states(self, n_steps, rng, start_state=None):
        observations = np.zeros((n_steps, self.emission_model.embedded_dimension))
        states = np.zeros(n_steps, dtype=int)

        if start_state is None:
            states[0] = rng.choice(self.transition_matrix.shape[0], p=self.initial_distribution)
        else:
            states[0] = start_state

        observations[0, :] = self.emission_model.get_observation(states[0], rng)

        for t in range(1, n_steps):
            if self.mutation_function is not None:
                transition_matrix, emission_model, initial_distribution = self.mutation_function(self, t, rng)
                self.transition_matrix = transition_matrix
                self.emission_model = emission_model
                self.initial_distribution = initial_distribution

            pvec = self.transition_matrix[states[t - 1], :]
            states[t] = rng.choice(self.transition_matrix.shape[0], p=pvec)

            observations[t, :] = self.emission_model.get_observation(states[t], rng)
        return states, observations

    def advance_one_step(self, rng, old_state):
        pvec = self.transition_matrix[old_state, :]
        new_state = rng.choice(self.transition_matrix.shape[0], p=pvec)
        new_observation = self.emission_model.get_observation(new_state, rng)
        return new_state, new_observation

    def simulate(self, n_steps, rng):
        return self.simulate_with_states(n_steps, rng)[1]