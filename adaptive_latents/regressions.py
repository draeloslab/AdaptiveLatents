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
        """This function saves an observation and possibly updates initializes parameters if the regressor has seen
        enough data."""

    @abstractmethod
    def predict(self, x):
        """This function returns the predicted y for some given x. It might return nans if there aren't enough observations yet."""

    # @abstractmethod
    # def _observe(self, x, y):
    #     """This function observes a datapoint, but does not check if the regression is initialized; usually you want
    #     safe_observe."""

    # @abstractmethod
    # def initialize(self, use_stored=True, x_history=None, y_history=None):
    #     """This is called when the algorithm has enough information to start making predictions; it initializes the
    #     regression. Usually use_stored will be true, and the algorithm will use previously observed data, but x and y
    #     can also be passed in as the initialization data. This will often be called by safe_observe"""


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

    def initialize(self, use_stored=True, x_history=None, y_history=None):
        if not use_stored:
            for i in np.arange(x_history.shape[0]):
                self._observe(x=x_history[i], y=y_history[i], update_D=False)
        self.D = np.linalg.pinv(self.F)

    def _observe(self, x, y, update_D=False):
        x = x.reshape([-1, 1])
        if self.add_intercept:
            x = np.vstack([x,[1]])
        y = np.squeeze(y)

        if update_D:
            self.D = rank_one_update_formula1(self.D, x, x)
        else:
            self.F = self.F + x @ x.T
        self.c = self.c + x * y

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
                self.initialize()

    def get_beta(self):
        if self.D is None:
            return np.zeros((self.input_d, self.output_d)) * np.nan
        return self.D @ self.c

    def predict(self, x):
        if self.D is None:
            return np.nan * np.ones(shape=[self.output_d, ])

        x = x.reshape([-1,1])
        if self.add_intercept:
            x = np.vstack([x,[1]])

        w = self.D @ self.c

        return x.T @ w


class SemiRegularizedRegressor(OnlineRegressor):
    def __init__(self, input_d, output_d, regularization_factor=0.01, add_intercept=True):
        super().__init__(input_d + add_intercept, output_d)
        self.add_intercept = add_intercept

        # core stuff
        self.D = np.eye(self.input_d)/regularization_factor
        self.c = np.zeros([self.input_d, self.output_d])

        self.n_observed = 0

    def _observe(self, x, y):
        x = x.reshape([-1, 1])
        if self.add_intercept:
            x = np.vstack([x,[1]])
        y = np.squeeze(y)

        self.D = rank_one_update_formula1(self.D, x, x)
        self.c = self.c + x * y

        self.n_observed += 1

    def observe(self, x, y):
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            return
        x, y = np.array(x), np.array(y)

        self._observe(x, y)

    def get_beta(self, exclude_intercept=False):
        if not exclude_intercept:
            beta = self.D @ self.c
        else:
            beta = (self.D @ self.c)[:-1, :]
        return beta

    def predict(self, x):
        x = x.reshape([-1,1])
        if self.add_intercept:
            x = np.vstack([x,[1]])
        return np.squeeze(x.T @ self.get_beta())


class SymmetricNoisyRegressor(OnlineRegressor):
    def __init__(self, input_d, output_d, forgetting_factor=1e-4, noise_scale=1e-3, n_perturbations=3, seed=24,
                 init_min_ratio=3, add_intercept=True):
        super().__init__(input_d + add_intercept, output_d)
        # todo: set a max init len
        self.add_intercept = add_intercept

        forgetting_factor = 1-forgetting_factor

        if n_perturbations < 1:
            raise Exception("the number of perturbations has to be more than 1")

        if forgetting_factor > 1:
            raise Exception("the forgetting factor should be in (0,1]")

        # core stuff
        self.forgetting_factor = forgetting_factor
        self.D = None
        self.F = np.zeros([self.input_d, self.input_d])
        self.c = np.zeros([self.input_d, self.output_d])
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)
        self.n_perturbations = n_perturbations

        # initializations
        self.init_min_ratio = init_min_ratio
        self.n_observed = 0

    def initialize(self, use_stored=True, x_history=None, y_history=None):
        if not use_stored:
            for i in np.arange(x_history.shape[0]):
                self._observe(x=x_history[i], y=y_history[i], update_D=False)
        self.D = np.linalg.pinv(self.F)

    def _observe(self, x, y, update_D=False):
        x = x.reshape([-1, 1])
        if self.add_intercept:
            x = np.vstack([x,[1]])
        y = np.squeeze(y)
        if update_D:
            self.D /= self.forgetting_factor
        else:
            self.F *= self.forgetting_factor

        self.c *= self.forgetting_factor

        for _ in range(self.n_perturbations):
            dx = self.rng.normal(scale=self.noise_scale, size=x.shape)
            for c in [-1, 1]:
                new_x = x + dx * c
                if update_D:
                    self.D = rank_one_update_formula1(self.D, new_x, new_x)
                else:
                    self.F = self.F + new_x @ new_x.T
                self.c = self.c + new_x * y

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
                self.initialize()


    def get_beta(self, exclude_intercept=False):
        if self.D is None:
            return np.zeros((self.input_d, self.output_d)) * np.nan
        if not exclude_intercept:
            beta = self.D @ self.c
        else:
            beta = (self.D @ self.c)[:-1, :]
        return beta

    def predict(self, x):
        if self.D is None:
            return np.nan * np.ones(shape=[self.output_d, ])
        if self.add_intercept:
            x = np.vstack([x.reshape((-1,1)), [1]])
        return x.T @ self.get_beta()


# class WindowRegressor(OnlineRegressor):
#     def __init__(self, input_d, output_d, window_size=1_000, init_min_ratio=3):
#         super().__init__(input_d, output_d)
#         assert False, "add an add_intercept"
#
#         # core stuff
#         self.window_size = window_size
#
#         self.x_window = []
#         self.y_window = []
#         self.D = None
#         self.F = np.zeros([self.input_d, self.input_d])
#         self.c = np.zeros([self.input_d, self.output_d])
#
#         # initializations
#         self.init_min_ratio = init_min_ratio
#         self.n_observed = 0
#
#     def initialize(self, use_stored=True, x_history=None, y_history=None):
#         if not use_stored:
#             for i in np.arange(x_history.shape[0]):
#                 self._observe(x=x_history[i], y=y_history[i], update_D=False)
#
#         self.D = np.linalg.pinv(self.F)
#
#     def _observe(self, x, y, update_D=False):
#         # x and y should not be multiple time-steps big
#         x = x.reshape([-1, 1])
#         y = np.squeeze(y)
#
#         # trim windows
#         assert len(self.x_window) == len(self.y_window)
#         while len(self.x_window) > self.window_size:
#             removed_x = self.x_window.pop(0)
#             removed_y = self.y_window.pop(0)
#             if update_D:
#                 self.D = rank_one_update_formula1(self.D, removed_x, -removed_x)
#             else:
#                 self.F -= removed_x @ removed_x.T
#             self.c -= removed_x @ removed_y.reshape([-1, self.output_d])
#
#         self.x_window.append(x)
#         self.y_window.append(y)
#
#         # update c and D
#         if update_D:
#             self.D = rank_one_update_formula1(self.D, x, x)
#         else:
#             self.F += x @ x.T
#         self.c += x @ y.reshape([-1, self.output_d])
#
#         self.n_observed += 1
#
#     def observe(self, x, y):
#         if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
#             return
#         x, y = np.array(x), np.array(y)
#         if self.n_observed >= self.init_min_ratio * self.input_d or self.D is not None:
#             self._observe(x, y, update_D=True)
#         else:
#             self._observe(x, y, update_D=False)
#             if self.n_observed >= self.init_min_ratio * self.input_d:
#                 self.initialize()
#
#     def predict(self, x):
#         if self.D is None:
#             return np.nan * np.ones(shape=[self.output_d, ])
#
#         w = self.D @ self.c
#         return x.T @ w


class NearestNeighborRegressor(OnlineRegressor):
    def __init__(self, input_d, output_d, maxlen=1_000):
        super().__init__(input_d, output_d)
        self.maxlen = maxlen
        self.history = np.zeros(shape=(maxlen, input_d + output_d)) * np.nan

        # index is the next row to write to, increases, and wraps
        self.index = 0

    def initialize(self, use_stored=True, x_history=None, y_history=None):
        if use_stored:
            pass
        else:
            assert len(x_history) == len(y_history)
            # todo: this can be optimized
            for i in range(len(x_history)):
                self._observe(x_history[i], y_history[i])

    def observe(self, x, y):
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            return
        self._observe(x, y)

    def _observe(self, x, y):
        self.history[self.index, :self.input_d] = x
        self.history[self.index, self.input_d:] = y
        self.index += 1
        if self.index >= self.history.shape[0]:
            self.index = 0

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
            super().__init__(input_d+output_d*n_steps, output_d, **kwargs)
            self._y_history = deque(maxlen=n_steps)

        def observe(self, x, y):
            self._y_history.append(y)

            if autoregress_only:
                x = 0*x

            if len(self._y_history) == self._y_history.maxlen:
                super().observe(np.hstack([np.array(self._y_history).flatten(), x]), y)

        def predict(self, x):
            if autoregress_only:
                x = 0*x

            if len(self._y_history) == self._y_history.maxlen:
                return super().predict(np.hstack([np.array(self._y_history).flatten(), x]))
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
