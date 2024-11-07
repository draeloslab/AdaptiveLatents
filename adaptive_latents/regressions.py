import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from .transformer import DecoupledTransformer
from .timed_data_source import ArrayWithTime
import jax


@jax.jit
def rank_one_update_formula1(D, x1, x2):
    # TODO: maybe this is only faster if we put it on the GPU? maybe move the data?
    return D - (D @ x1 @ x2.T @ D) / (1 + x2.T @ D @ x1)


class OnlineRegressor(ABC):
    @abstractmethod
    def observe(self, x, y):
        """
        This function saves an observation and possibly updates initializes parameters if the regressor has seen
        enough data.
        Inputs should be 1d?
        """

    @abstractmethod
    def predict(self, x):
        """
        This function returns the predicted y for some given x. It might return nans if there aren't enough observations yet.
        It predicts for one x at a time, so the outputs are 1d.
        """


class BaseVanillaOnlineRegressor(OnlineRegressor):
    def __init__(self, init_min_ratio=1.1, add_intercept=True, regularization_factor=0.01):
        self.add_intercept = add_intercept
        self.init_min_ratio = init_min_ratio
        self.regularization_factor = regularization_factor

        # core stuff
        self.input_d = None
        self.output_d = None
        self.D = None  # this should be None for a while
        self.F = None
        self.c = None

        # initializations
        self.n_observed = 0

    def format_x(self, x):
        x = x.reshape([-1, 1])
        if self.add_intercept:
            x = np.vstack([x, [1]])
        return x

    def _observe(self, x, y, update_D=False):
        x = self.format_x(x)
        y = np.squeeze(y)

        if update_D:
            self.D = rank_one_update_formula1(self.D, x, x)
        else:
            self.F = self.F + x @ x.T
        self.c = self.c + x*y

        self.n_observed += 1

    def observe(self, x, y):
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            return

        # x and y should be vectors
        if self.F is None and self.c is None:  # this is the first observation
            self.input_d = x.size + self.add_intercept
            self.output_d = y.size
            if self.regularization_factor == 0:
                self.F = np.zeros([self.input_d, self.input_d])
                self.c = np.zeros([self.input_d, self.output_d])
            else:
                self.D = np.eye(self.input_d) / self.regularization_factor
                self.c = np.zeros([self.input_d, self.output_d])

        if self.n_observed >= self.init_min_ratio * self.input_d or self.D is not None:
            self._observe(x, y, update_D=True)
        else:
            self._observe(x, y, update_D=False)
            if self.n_observed >= self.init_min_ratio * self.input_d:
                # initialize
                self.D = np.linalg.pinv(self.F)

    def get_beta(self):
        if self.c is None:
            return np.nan

        if self.D is None:
            return np.zeros((self.input_d, self.output_d)) * np.nan
        return self.D @ self.c

    def predict(self, x):
        if self.c is None:
            return np.array(np.nan)

        x = self.format_x(x)
        beta = self.get_beta()

        return (x.T @ beta).flatten()

    # def project_input(self, x):
    #     if self.c is None:
    #         return np.array(np.nan)
    #
    #     x = self.format_x(x)
    #     beta = self.get_beta()
    #
    #     u, s, vh = np.linalg.svd(beta)
    #     return (x.T @ u).flatten()


class BaseNearestNeighborRegressor(OnlineRegressor):
    def __init__(self, maxlen=1_000):
        super().__init__()
        self.maxlen = maxlen
        self.output_d = None
        self.input_d = None
        self.history = None

        # index is the next row to write to, increases, and wraps
        self.index = 0

    def observe(self, x, y):
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            return

        if self.history is None:
            self.input_d = x.size
            self.output_d = y.size
            self.history = np.zeros(shape=(self.maxlen, self.input_d + self.output_d)) * np.nan
        self._observe(x, y)

    def _observe(self, x, y):
        self.history[self.index, :self.input_d] = x
        self.history[self.index, self.input_d:] = y
        self.index = (self.index + 1) % self.maxlen

    def predict(self, x):
        if self.history is None:
            return np.nan
        distances = np.linalg.norm(self.history[:, :self.input_d] - np.squeeze(x), axis=1)
        try:
            idx = np.nanargmin(distances)
        except ValueError:
            return np.nan * np.empty(shape=(self.output_d,))
        return self.history[idx, self.input_d:]


def auto_regression_decorator(regressor_class: OnlineRegressor, n_steps=1, autoregress_only=False):
    class AutoRegressor(regressor_class):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._y_history = deque(maxlen=n_steps)

        def observe(self, x, y):
            self._y_history.append(y)

            if autoregress_only:
                x = 0 * x

            if len(self._y_history) == self._y_history.maxlen:
                super().observe(np.hstack([np.array(self._y_history).flatten(), x.flatten()]), y)

        def predict(self, x):
            if autoregress_only:
                x = 0 * x

            if len(self._y_history) == self._y_history.maxlen:
                return super().predict(np.hstack([np.array(self._y_history).flatten(), x.flatten()]))
            else:
                return np.array([[np.nan]])

    return AutoRegressor


class VanillaOnlineRegressor(DecoupledTransformer, BaseVanillaOnlineRegressor):
    base_algorithm = BaseVanillaOnlineRegressor

    def __init__(self, input_streams=None, **kwargs):
        input_streams = input_streams or {0: 'X', 1: 'Y'}
        super().__init__(input_streams=input_streams,**kwargs)
        self.log = {'preq_error':[], 't': []}
        self.last_seen = {}

    def partial_fit(self, data, stream=0):
        if self.frozen:
            return
        self.pre_log_for_partial_fit(data, stream)
        self._partial_fit(data, stream)
        self.log_for_partial_fit(data, stream)

    def pre_log_for_partial_fit(self, data, stream):
        if self.log_level > 0:
            stream_label = self.input_streams[stream]
            if stream_label in ('X', 'Y'):
                if np.isnan(data).any():
                    return

                last_seen = dict(self.last_seen)
                last_seen[stream_label] = data
                if len(last_seen) == 2:
                    for i in range(last_seen['X'].shape[0]):
                        pred = self.predict(last_seen['X'][i])
                        self.log['preq_error'].append(pred - last_seen['Y'][i])
                        if isinstance(last_seen['X'], ArrayWithTime):
                            self.log['t'].append(max(last_seen['X'].t, last_seen['Y'].t))


    def get_params(self, deep=True):
        return dict(init_min_ratio=self.init_min_ratio, add_intercept=self.add_intercept, regularization_factor=self.regularization_factor) | super().get_params()

    def _partial_fit(self, data, stream=0):
        stream_label = self.input_streams[stream]
        if stream_label in ('X', 'Y'):
            if np.isnan(data).any():
                return

            self.last_seen[stream_label] = data
            if len(self.last_seen) == 2:
                for i in range(self.last_seen['X'].shape[0]):
                    self.observe(self.last_seen['X'][i], self.last_seen['Y'][i])
                self.last_seen = {}

    def transform(self, data, stream=0, return_output_stream=False):
        stream_label = self.input_streams[stream]
        if stream_label in {'X', 'qX'}:
            if np.isnan(data).any():
                data = np.nan * data
            else:
                prediction = [self.predict(row) for row in data]
                if isinstance(data, ArrayWithTime):
                    data = ArrayWithTime(prediction, data.t)
                else:
                    data = np.array(data)

        stream = self.output_streams[stream]

        return data, stream if return_output_stream else data

    def plot_preq_error(self, ax):
        t = np.array(self.log['t'])
        preq_error = np.array(self.log['preq_error'])
        sq_error = preq_error**2
        ax.plot(t, sq_error)
        ax.set_xlabel('time')
        ax.set_ylabel('regression training preqential error')
