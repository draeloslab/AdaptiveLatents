import numpy as np
from abc import ABC, abstractmethod
from collections import deque


def rank_one_update_formula1(D, x1, x2=None):
    if x2 is None:
        x2 = x1
    return D - (D @ x1 @ x2.T @ D) / (1 + x2.T @ D @ x1)


class OnlineRegressor(ABC):
    def __init__(self, input_d, output_d):
        self.input_d = input_d
        self.output_d = output_d

    @abstractmethod
    def initialize(self, use_stored=True, x_history=None, y_history=None):
        """This is called when the algorithm has enough information to start making predictions; it initializes the
        regression. Usually use_stored will be true, and the algorithm will use previously observed data, but x and y
        can also be passed in as the initialization data."""
        pass

    @abstractmethod
    def safe_observe(self, x, y):
        """This function saves an observation and possibly updates initializes parameters if the regressor has seen
        enough data."""
        pass

    @abstractmethod
    def observe(self, x, y):
        """This function observes a datapoint; usually you want safe_observe."""
        pass

    @abstractmethod
    def predict(self, x):
        pass


class VanillaOnlineRegressor(OnlineRegressor):
    def __init__(self, input_d, output_d, init_min_ratio=3):
        super().__init__(input_d, output_d)

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
                self.observe(x=x_history[i], y=y_history[i], update_D=False)
        self.D = np.linalg.pinv(self.F)

    def observe(self, x, y, update_D=False):
        x = x.reshape([-1, 1])
        y = np.squeeze(y)

        if update_D:
            self.D = rank_one_update_formula1(self.D, x)
        else:
            self.F = self.F + x @ x.T
        self.c = self.c + x * y

        self.n_observed += 1

    def safe_observe(self, x, y):
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            return
        x, y = np.array(x), np.array(y)
        if self.n_observed >= self.init_min_ratio * self.input_d or self.D is not None:
            self.observe(x, y, update_D=True)
        else:
            self.observe(x, y, update_D=False)
            if self.n_observed >= self.init_min_ratio * self.input_d:
                self.initialize()

    def get_beta(self):
        if self.D is None:
            return np.zeros((self.input_d, self.output_d)) * np.nan
        return self.D @ self.c

    def predict(self, x):
        if self.D is None:
            return np.nan * np.ones(shape=[self.output_d, ])

        w = self.D @ self.c

        return x.T @ w



class SymmetricNoisyRegressor(OnlineRegressor):
    def __init__(self, input_d, output_d, forgetting_factor=1e-4, noise_scale=1e-3, n_perturbations=3, seed=24,
                 init_min_ratio=3):
        super().__init__(input_d, output_d)

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
                self.observe(x=x_history[i], y=y_history[i], update_D=False)
        self.D = np.linalg.pinv(self.F)

    def observe(self, x, y, update_D=False):
        x = x.reshape([-1, 1])
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
                    self.D = rank_one_update_formula1(self.D, new_x)
                else:
                    self.F = self.F + new_x @ new_x.T
                self.c = self.c + new_x * y

        self.n_observed += 1

    def safe_observe(self, x, y):
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            return
        x, y = np.array(x), np.array(y)
        if self.n_observed >= self.init_min_ratio * self.input_d or self.D is not None:
            self.observe(x, y, update_D=True)
        else:
            self.observe(x, y, update_D=False)
            if self.n_observed >= self.init_min_ratio * self.input_d:
                self.initialize()

    def get_beta(self):
        return self.D @ self.c

    def predict(self, x):
        if self.D is None:
            return np.nan * np.ones(shape=[self.output_d, ])

        w = self.D @ self.c

        return x.T @ w

class HistoricalSNR(OnlineRegressor):
    def __init__(self, input_d, output_d, forgetting_factor=1e-4, noise_scale=1e-3, n_perturbations=3, seed=24,
                 init_min_ratio=3, history_length=1):
        super().__init__(input_d, output_d)

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
        self.x_history = deque(maxlen=history_length)
        self.y_history = deque(maxlen=history_length)

        # initializations
        self.init_min_ratio = init_min_ratio
        self.n_observed = 0

    def initialize(self, use_stored=True, x_history=None, y_history=None):
        if not use_stored:
            for i in np.arange(x_history.shape[0]):
                self.observe(x=x_history[i], y=y_history[i], update_D=False)
        self.D = np.linalg.pinv(self.F)

    def observe(self, x, y, update_D=False):
        x = x.reshape([-1, 1])
        y = np.squeeze(y)
        self.x_history.append(x)
        self.y_history.append(y)

        x = np.array(self.x_history).flatten().reshape([-1, 1])
        y = np.array(self.y_history).flatten()

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
                    self.D = rank_one_update_formula1(self.D, new_x)
                else:
                    self.F = self.F + new_x @ new_x.T
                self.c = self.c + new_x * y

        self.n_observed += 1

    def safe_observe(self, x, y):
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            return
        x, y = np.array(x), np.array(y)
        if self.n_observed >= self.init_min_ratio * self.input_d or self.D is not None:
            self.observe(x, y, update_D=True)
        else:
            self.observe(x, y, update_D=False)
            if self.n_observed >= self.init_min_ratio * self.input_d:
                self.initialize()

    def get_beta(self):
        return self.D @ self.c

    def predict(self, x):
        x = np.array(list(self.x_history)[:-1] + [x])
        x = x.flatten()
        if self.D is None:
            return np.nan * np.ones(shape=[self.output_d, ])

        w = self.D @ self.c

        return x.T @ w



class BinarySNR(SymmetricNoisyRegressor):
    def predict(self,x):
        return np.round(super().predict(x))


# class SlowWindowRegressor(OnlineRegressor):
#     def __init__(self, N,  window_size, forgetting_factor=1, init_min_ratio=3):
#         super().__init__()
#
#         if forgetting_factor > 1:
#             raise Exception("the forgetting factor should be in (0,1]")
#
#         # core stuff
#         self.N = N
#         self.forgetting_factor = forgetting_factor
#         self.window_size = window_size
#
#         self.x_window = []
#         self.y_window = []
#         self.D = None
#         self.c = np.zeros([N, 1])
#
#
#         # initializations
#         self.init_min_ratio = init_min_ratio
#         self.n_observed = 0
#
#     def initialize(self, use_stored=True, x=None, y=None):
#         if not use_stored:
#             for i in np.arange(x.shape[0]):
#                 self.update(x=x[i], y=y[i], update_D=False)
#
#         x_mat = np.array(self.x_window).reshape([-1,self.N])
#         self.D = np.linalg.pinv(x_mat.T @ x_mat)
#
#     def update(self, x, y, update_D=False):
#         # x and y should not be multiple time-steps big
#         x = x.reshape([-1,1])
#         y = np.squeeze(y)
#
#         # trim windows
#         assert len(self.x_window) == len(self.y_window)
#         while len(self.x_window) > self.window_size:
#             self.x_window.pop(0)
#             self.y_window.pop(0)
#
#         # enforce forgetting factor
#         if self.forgetting_factor < 1:
#             for i in range(len(self.x_window)):
#                 # TODO: make all the forgetting factors uniform (with like square roots or whatever)
#                 self.x_window[i] *= self.forgetting_factor
#                 self.y_window[i] *= self.forgetting_factor
#
#         self.x_window.append(x)
#         self.y_window.append(y)
#
#         # update c and D
#         # todo: think about this D and update_D decision more
#         x_mat = np.array(self.x_window).reshape([-1,self.N])
#         y_mat = np.array(self.y_window).reshape([-1,1])
#         if update_D:
#             self.D = np.linalg.pinv(x_mat.T @ x_mat)
#         self.c = x_mat.T @ y_mat
#
#         self.n_observed += 1
#
#     def lazy_observe(self, x, y):
#         x, y = np.array(x), np.array(y)
#         if self.n_observed >= self.init_min_ratio * self.N or self.D is not None:
#             self.update(x, y, update_D=True)
#         else:
#             self.update(x, y, update_D=False)
#             if self.n_observed >= self.init_min_ratio * self.N:
#                 self.initialize()
#
#     def predict(self, x):
#         if self.D is None:
#             return np.nan * np.ones(shape=[self.n_obs,])
#
#         w = self.D @ self.c
#         return np.squeeze(x.T @ w)


class WindowRegressor(OnlineRegressor):
    def __init__(self, input_d, output_d, window_size=1_000, init_min_ratio=3):
        super().__init__(input_d, output_d)

        # core stuff
        self.window_size = window_size

        self.x_window = []
        self.y_window = []
        self.D = None
        self.F = np.zeros([self.input_d, self.input_d])
        self.c = np.zeros([self.input_d, self.output_d])

        # initializations
        self.init_min_ratio = init_min_ratio
        self.n_observed = 0

    def initialize(self, use_stored=True, x_history=None, y_history=None):
        if not use_stored:
            for i in np.arange(x_history.shape[0]):
                self.observe(x=x_history[i], y=y_history[i], update_D=False)

        self.D = np.linalg.pinv(self.F)

    def observe(self, x, y, update_D=False):
        # x and y should not be multiple time-steps big
        x = x.reshape([-1, 1])
        y = np.squeeze(y)

        # trim windows
        assert len(self.x_window) == len(self.y_window)
        while len(self.x_window) > self.window_size:
            removed_x = self.x_window.pop(0)
            removed_y = self.y_window.pop(0)
            if update_D:
                self.D = rank_one_update_formula1(self.D, removed_x, -removed_x)
            else:
                self.F -= removed_x @ removed_x.T
            self.c -= removed_x @ removed_y.reshape([-1, self.output_d])

        self.x_window.append(x)
        self.y_window.append(y)

        # update c and D
        if update_D:
            self.D = rank_one_update_formula1(self.D, x)
        else:
            self.F += x @ x.T
        self.c += x @ y.reshape([-1, self.output_d])

        self.n_observed += 1

    def safe_observe(self, x, y):
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            return
        x, y = np.array(x), np.array(y)
        if self.n_observed >= self.init_min_ratio * self.input_d or self.D is not None:
            self.observe(x, y, update_D=True)
        else:
            self.observe(x, y, update_D=False)
            if self.n_observed >= self.init_min_ratio * self.input_d:
                self.initialize()

    def predict(self, x):
        if self.D is None:
            return np.nan * np.ones(shape=[self.output_d, ])

        w = self.D @ self.c
        return x.T @ w


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
                self.observe(x_history[i], y_history[i])

    def safe_observe(self, x, y):
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            return
        self.observe(x, y)

    def observe(self, x, y):
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


class AutoRegressor(OnlineRegressor):
    def __init__(self, input_d, output_d):
        super().__init__(input_d, output_d)
        self.history = None

    def safe_observe(self, x, y):
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            return
        self.observe(x,y)

    def observe(self, x, y):
        self.history = y

    def predict(self, x):
        if self.history is None:
            return np.nan * np.empty(shape=(self.output_d,))

        return self.history

    def initialize(self, use_stored=True, x_history=None, y_history=None):
        if use_stored:
            pass
        else:
            self.history = y_history[-1]


"""
ideas:
    periodically recalculate D and c
    zero out small elements of D and c
    threshold D and c values?
    constant term and re-weighting
    prior on constant term
    force the constant term to be the mean
"""
