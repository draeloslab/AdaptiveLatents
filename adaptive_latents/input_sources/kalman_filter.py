import warnings

import numpy as np
from scipy.stats import multivariate_normal

from adaptive_latents.predictor import Predictor


class KalmanFilter:
    # TODO: make this a transformer once fit
    def __init__(self, use_steady_state_k=False, subtract_means=True):
        self.use_steady_state_K = use_steady_state_k
        self.subtract_means = subtract_means

        self.A = None  # state transitions
        self.C = None  # link between states and observations
        self.W = None  # state noise
        self.Q = None  # observation noise

        self.X_mean = None
        self.Y_mean = None

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
            if max([len(x) for x in _X]) == 1:
                warnings.warn("not fitting because there isn't enough data")
                return
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

    @staticmethod
    def inference_step(state, state_var, *, A, C, W, Q, X_mean, Y_mean, Y=None, kalman_gain=None):
        state = state - X_mean
        state = state @ A
        state_var = A @ state_var @ A.T + W

        if Y is not None:
            Y = Y - Y_mean
            if kalman_gain is None:
                kalman_gain = state_var @ C @ np.linalg.pinv(C.T @ state_var @ C + Q)
            state = state + (Y - state @ C) @ kalman_gain.T
            state_var = (np.eye(C.shape[0]) - C @ kalman_gain.T) @ state_var

        return state + X_mean, state_var


    def step(self, Y=None):
        self.state, self.state_var = self.inference_step(self.state, self.state_var, Y=Y, A=self.A, C=self.C, W=self.W, Q=self.Q, Y_mean=self.Y_mean, X_mean=self.X_mean, kalman_gain=None if not self.use_steady_state_K else self.steady_state_K)
        return self.state

    def predict_state_and_var(self, n_steps, state, state_var):
        prediction = np.zeros((n_steps+1, self.A.shape[0])) * np.nan
        prediction_var = np.zeros((n_steps+1, self.A.shape[0], self.A.shape[0])) * np.nan
        prediction[0] = state
        prediction_var[0] = state_var
        for i in range(n_steps):
            state, state_var = self.inference_step(state, state_var, Y=None, A=self.A, C=self.C, W=self.W, Q=self.Q, Y_mean=self.Y_mean, X_mean=self.X_mean, kalman_gain=None if not self.use_steady_state_K else self.steady_state_K)
            prediction[i+1] = state
            prediction_var[i+1] = state_var

        return prediction, prediction_var

    def predict(self, n_steps, initial_state=None, initial_state_var=None):
        state = initial_state if initial_state is not None else self.state
        state_var = initial_state_var if initial_state_var is not None else self.state_var
        prediction, prediction_var = self.predict_state_and_var(n_steps, state, state_var)
        return prediction


class StreamingKalmanFilter(Predictor, KalmanFilter):
    base_algorithm = KalmanFilter
    def __init__(self, *, steps_between_refits = 25, use_steady_state_k=False, subtract_means=True, no_hidden_state=True, input_streams=None, output_streams=None, log_level=None, check_dt=False, n_steps_to_predict=1, max_history_length=5000):
        input_streams = input_streams or {0: 'X', 1: 'Y', 2: 'dt_X', 'toggle_parameter_fitting': 'toggle_parameter_fitting'}
        Predictor.__init__(self, input_streams=input_streams, output_streams=output_streams, log_level=log_level, check_dt=check_dt, n_steps_to_predict=n_steps_to_predict)
        KalmanFilter.__init__(self, use_steady_state_k=use_steady_state_k, subtract_means=subtract_means)
        self.no_hidden_state = no_hidden_state
        self.steps_between_refits = steps_between_refits
        self.max_history_length = max_history_length

        self.last_seen = {}
        self.latent_state_history = [[]]
        self.observation_history = [[]]

    def predict(self, n_steps):
        if self.A is not None:
            predicted_latent_state = KalmanFilter.predict(self, n_steps)[-1]
            predicted_observation = (predicted_latent_state @ self.C)
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

            assert len(self.latent_state_history[-1]) == len(self.observation_history[-1])
            n_seen = sum(len(x) if len(x) > 1 else 0 for x in self.observation_history)
            if (
                    n_seen % self.steps_between_refits == 0
                    and len(self.observation_history[-1]) > 1
                    and self.parameter_fitting
            ):
                self.fit(X=self.latent_state_history, Y=self.observation_history)
                latent = np.squeeze(self.latent_state_history[-1])
                obs = np.squeeze(self.observation_history[-1])

                while sum([len(x) for x in self.observation_history]) > self.max_history_length:
                    if len(self.observation_history[0]) == 2:
                        self.observation_history.pop(0)
                        self.latent_state_history.pop(0)
                    else:
                        self.observation_history[0].pop(0)
                        self.latent_state_history[0].pop(0)


                constant = min(self.steps_between_refits, len(obs)) # TODO: set this more rigorously
                self.state = latent[obs.shape[0]-constant]
                for i in range(constant):
                    self.step(Y=obs[obs.shape[0]-constant+i])

    def toggle_parameter_fitting(self, value=None):
        before = self.parameter_fitting
        super().toggle_parameter_fitting(value)
        if before and not self.parameter_fitting:
            self.last_seen = {}
            if len(self.latent_state_history[-1]) > 1:
                self.latent_state_history.append([])
            else:
                self.latent_state_history[-1] = []

            if len(self.observation_history[-1]) > 1:
                self.observation_history.append([])
            else:
                self.observation_history[-1] = []

    def get_state(self):
        state = self.state if self.state is not None else np.array([np.nan])
        return state

    def get_params(self, deep=True):
        return super().get_params(deep) | dict(use_steady_state_k=self.use_steady_state_K, subtract_means=self.subtract_means, steps_between_refits=self.steps_between_refits)

    def get_arbitrary_dynamics_parameter(self):
        if self.A is None:
            return np.nan

        return self.A


    def unevaluated_log_pred_p(self, n_steps):
        if self.A is None:
            return lambda x: np.nan

        evals, evecs  = np.linalg.eigh(self.state_var)
        evals = np.abs(evals)  # TODO: this should not be necessary
        state_var = (evecs * evals) @ evecs.T

        state = np.array(self.state)
        A = np.array(self.A)
        C = np.array(self.C)
        W = np.array(self.W)
        Q = np.array(self.Q)
        X_mean = np.array(self.X_mean)
        Y_mean = np.array(self.Y_mean)

        inner_state = state
        inner_state_var = state_var
        for i in range(n_steps):
            inner_state, inner_state_var = KalmanFilter.inference_step(inner_state, inner_state_var, A=A, C=C, W=W, Q=Q, X_mean=X_mean, Y_mean=Y_mean)
        try:
            rv = multivariate_normal(mean=inner_state.flatten(), cov=inner_state_var)
        except np.linalg.LinAlgError:
            warnings.warn("covariance matrix is not positive definite")
            return lambda x: np.nan

        return rv.logpdf