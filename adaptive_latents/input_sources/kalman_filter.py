import numpy as np
from adaptive_latents.transformer import StreamingTransformer, ArrayWithTime
from adaptive_latents.predictor import Predictor


class KalmanFilter:
    def __init__(self, use_steady_state_k=False, subtract_means=True):
        self.use_steady_state_K = use_steady_state_k
        self.subtract_means = subtract_means

        self.A = None  # state transitions
        self.C = None  # link between states and observations
        self.W = None  # state noise
        self.Q = None  # observation noise

        self.steady_state_K = None

        self.state_var = None
        self.state = None

    def fit(self, X, Y):
        _X = None
        if isinstance(X, list):
            assert len(np.array(X[0]).shape) == 3
            _X = X
            X, Y = np.vstack([np.vstack(x) for x in X]), np.vstack([np.vstack(y) for y in Y])

        X_mean, Y_mean = (X.mean(axis=0), Y.mean(axis=0)) if self.subtract_means else (0,0)

        X = X - X_mean
        Y = Y - Y_mean

        origin = X[:-1]
        destination = X[1:]
        if _X is not None:
            origin = np.vstack([np.vstack(x[:-1]) for x in _X])
            destination = np.vstack([np.vstack(x[1:]) for x in _X])
        A, _, _, _ = np.linalg.lstsq(origin, destination)

        C, _, _, _ = np.linalg.lstsq(X, Y)

        w = X[1:] - X[:-1] @ A
        W = (w.T @ w) / (X.shape[1] - 1)

        q = Y - X @ C
        Q = (q.T @ q) / (X.shape[1])

        # model variables
        self.A = A
        self.C = C
        self.W = W
        self.Q = Q
        self.X_mean = X_mean
        self.Y_mean = Y_mean

        if self.use_steady_state_K:
            m = X.shape[1]
            P = W
            matrix = C.T @ P @ C + Q

            K_old = P @ C @ np.linalg.pinv(matrix)
            P = (np.eye(m) - C @ K_old.T) @ P
            for i in range(3000):
                P = A @ P @ A.T + W
                matrix = C.T @ P @ C + Q
                K = P @ C @ np.linalg.pinv(matrix)
                P = (np.eye(m) - C @ K.T) @ P

                dif = np.abs(K - K_old)
                K_old = K
                if (dif < 1E-16).all():
                    break
            self.steady_state_K = K

        # state variables
        self.state = np.zeros_like(X[-1:])
        self.state_var = self.W

    def step(self, Y=None):
        state = self.state @ self.A
        state_var = self.A @ self.state_var @ self.A.T + self.W

        if Y is not None:
            Y = Y - self.Y_mean
            if not self.use_steady_state_K:
                kalman_gain = state_var @ self.C @ np.linalg.pinv(self.C.T @ state_var @ self.C + self.Q)
            else:
                kalman_gain = self.steady_state_K
            state = state + (Y - state @ self.C) @ kalman_gain.T
            state_var = (np.eye(self.C.shape[0]) - self.C @ kalman_gain.T) @ state_var

        self.state = state
        self.state_var = state_var
        return state + self.X_mean

    def predict(self, n_steps, initial_state=None, initial_state_var=None):
        old_state, old_var = self.state, self.state_var  # TODO: I don't like saving the state like this

        if initial_state is not None:
            self.state = initial_state - self.X_mean
        if initial_state_var is None:
            self.state_var = self.state_var

        prediction = np.zeros((n_steps+1, self.A.shape[0])) * np.nan
        prediction[0] = self.state
        for i in range(n_steps):
            prediction[i+1,:] = self.step()

        self.state, self.state_var = old_state, old_var
        return prediction


class StreamingKalmanFilter(Predictor, KalmanFilter):
    base_algorithm = KalmanFilter
    def __init__(self, input_streams=None, output_streams=None, log_level=None, use_steady_state_k=False, subtract_means=True, no_hidden_state=True, check_dt=False):
        input_streams = input_streams or {0: 'X', 1: 'Y', 2: 'dt_X', 'toggle_parameter_fitting': 'toggle_parameter_fitting'}
        Predictor.__init__(self, input_streams=input_streams, output_streams=output_streams, log_level=log_level, check_dt=check_dt)
        KalmanFilter.__init__(self, use_steady_state_k=use_steady_state_k, subtract_means=subtract_means)
        self.no_hidden_state = no_hidden_state
        self.steps_between_refits = 25

        self.last_seen = {}
        self.latent_state_history = [[]]
        self.observation_history = [[]]

    def predict(self, n_steps):
        if self.A is not None:
            predicted_latent_state = KalmanFilter.predict(self, n_steps)
            predicted_observation = (predicted_latent_state @ self.C)[-1]
        else:
            predicted_observation = np.array([[np.nan]])
        return predicted_observation

    def observe(self, X, stream=None):
        semantic_stream = self.input_streams[stream]
        if semantic_stream in {'X', 'Y'}:
            if self.parameter_fitting:
                self.last_seen[semantic_stream] = X

            if ('Y' in self.last_seen or self.no_hidden_state) and 'X' in self.last_seen and self.parameter_fitting:
                self.observation_history[-1].append(self.last_seen['X'])
                self.latent_state_history[-1].append(self.last_seen['X' if self.no_hidden_state else 'Y'])

            if semantic_stream == 'X' and self.A is not None:
                self.step(X)

            if (
                    len(self.latent_state_history[-1]) == len(self.observation_history[-1])
                    and len(self.observation_history[-1])
                    and len(self.observation_history[-1]) % self.steps_between_refits == 0
            ):
                self.fit(X=self.latent_state_history, Y=self.observation_history)
                latent = np.squeeze(self.latent_state_history[-1])
                obs = np.squeeze(self.observation_history[-1])
                self.state = latent[obs.shape[0]-20]
                for i in range(20):
                    self.step(Y=obs[obs.shape[0]-20+i])

    def toggle_parameter_fitting(self, value=None):
        super().toggle_parameter_fitting(value)
        if not self.parameter_fitting:
            self.last_seen = {}
            if len(self.latent_state_history[-1]):
                self.latent_state_history.append([])
            if len(self.observation_history[-1]):
                self.observation_history.append([])

    def get_state(self):
        state = self.state if self.state is not None else np.array([np.nan])
        return state

    def get_params(self, deep=True):
        return super().get_params(deep) | dict(use_steady_state_k=self.use_steady_state_K, subtract_means=self.subtract_means)

    def get_arbitrary_dynamics_parameter(self):
        return self.A