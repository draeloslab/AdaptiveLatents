import numpy as np
from abc import ABC, abstractmethod
from collections import deque
import jax


@jax.jit
def rank_one_update_formula1(D, x1, x2):
    # TODO: maybe this is only faster if we put it on the GPU? maybe move the data?
    return D - (D @ x1 @ x2.T @ D) / (1 + x2.T @ D @ x1)


class OnlineRegressor(ABC):
    def __init__(self, input_d, output_d):
        self.input_d = input_d
        self.output_d = output_d

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


class VanillaOnlineRegressor(OnlineRegressor):
    def __init__(self, input_d, output_d, init_min_ratio=3, add_intercept=True):
        super().__init__(input_d + add_intercept, output_d)
        self.add_intercept = add_intercept

        # core stuff
        self.D = None
        self.F = np.zeros([self.input_d, self.input_d])
        self.c = np.zeros([self.input_d, self.output_d])

        # initializations
        self.init_min_ratio = init_min_ratio
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
        x, y = np.array(x), np.array(y)
        if self.n_observed >= self.init_min_ratio * self.input_d or self.D is not None:
            self._observe(x, y, update_D=True)
        else:
            self._observe(x, y, update_D=False)
            if self.n_observed >= self.init_min_ratio * self.input_d:
                # initialize
                self.D = np.linalg.pinv(self.F)

    def get_beta(self):
        if self.D is None:
            return np.zeros((self.input_d, self.output_d)) * np.nan
        return self.D @ self.c

    def predict(self, x):
        x = self.format_x(x)
        beta = self.get_beta()

        return (x.T @ beta).flatten()


class SemiRegularizedRegressor(VanillaOnlineRegressor):
    def __init__(self, input_d, output_d, add_intercept=True, regularization_factor=0.01):
        super().__init__(input_d, output_d, add_intercept=add_intercept)
        self.D = np.eye(self.input_d) / regularization_factor
        self.c = np.zeros([self.input_d, self.output_d])


class NearestNeighborRegressor(OnlineRegressor):
    def __init__(self, input_d, output_d, maxlen=1_000):
        super().__init__(input_d, output_d)
        self.maxlen = maxlen
        self.history = np.zeros(shape=(maxlen, input_d + output_d)) * np.nan

        # index is the next row to write to, increases, and wraps
        self.index = 0

    def observe(self, x, y):
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            return
        self._observe(x, y)

    def _observe(self, x, y):
        self.history[self.index, :self.input_d] = x
        self.history[self.index, self.input_d:] = y
        self.index = (self.index + 1) % self.maxlen

    def predict(self, x):
        distances = np.linalg.norm(self.history[:, :self.input_d] - np.squeeze(x), axis=1)
        try:
            idx = np.nanargmin(distances)
        except ValueError:
            return np.nan * np.empty(shape=(self.output_d,))
        return self.history[idx, self.input_d:]


def auto_regression_decorator(regressor_class: OnlineRegressor, n_steps=1, autoregress_only=False):
    class AutoRegressor(regressor_class):
        def __init__(self, input_d, output_d, **kwargs):
            super().__init__(input_d + output_d*n_steps, output_d, **kwargs)
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
                return np.nan * np.empty(shape=(self.output_d,))

    return AutoRegressor


"""
ideas:
    periodically recalculate D and c
    zero out small elements of D and c
    threshold D and c values?
    constant term and re-weighting
    prior on constant term
    force the constant term to be the mean
"""
