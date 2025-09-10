from abc import ABC, abstractmethod
from collections import deque
import copy
import warnings

import jax
from jax import numpy as jnp
import numpy

from .timed_data_source import ArrayWithTime
from .transformer import DecoupledTransformer


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
            x = numpy.vstack([x, [1]])
        return x

    def _observe(self, x, y, update_D=False):
        x = self.format_x(x)
        y = numpy.squeeze(y)

        if update_D:
            self.D = rank_one_update_formula1(self.D, x, x)
        else:
            self.F = self.F + x @ x.T
        self.c = self.c + x*y

        self.n_observed += 1

    def observe(self, x, y):
        if numpy.any(~numpy.isfinite(x)) or numpy.any(~numpy.isfinite(y)):
            return

        # x and y should be vectors
        if self.F is None and self.c is None:  # this is the first observation
            self.input_d = x.size + self.add_intercept
            self.output_d = y.size
            if self.regularization_factor == 0:
                self.F = numpy.zeros([self.input_d, self.input_d])
                self.c = numpy.zeros([self.input_d, self.output_d])
            else:
                self.D = numpy.eye(self.input_d) / self.regularization_factor
                self.c = numpy.zeros([self.input_d, self.output_d])

        if self.n_observed >= self.init_min_ratio * self.input_d or self.D is not None:
            self._observe(x, y, update_D=True)
        else:
            self._observe(x, y, update_D=False)
            if self.n_observed >= self.init_min_ratio * self.input_d:
                # initialize
                self.D = numpy.linalg.pinv(self.F)

    def get_beta(self):
        if self.c is None:
            return numpy.nan

        if self.D is None:
            return numpy.zeros((self.input_d, self.output_d)) * numpy.nan
        return self.D @ self.c

    def predict(self, x):
        if self.c is None:
            return numpy.array(numpy.nan)

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

class NonParametricRegressor(OnlineRegressor):
    def __init__(self, maxlen=1_000):
        super().__init__()
        self.maxlen = maxlen
        self.output_d = None
        self.input_d = None
        self.history = None

        # index is the next row to write to, increases, and wraps
        self.n_observed = 0

    def observe(self, x, y):
        if numpy.any(~numpy.isfinite(x)) or numpy.any(~numpy.isfinite(y)):
            return

        if self.history is None:
            self.input_d = x.size
            self.output_d = y.size
            self.history = numpy.zeros(shape=(self.maxlen, self.input_d + self.output_d))
            self.history[:,:self.input_d] = numpy.nan
        self._observe(x, y)

    def _observe(self, x, y):
        if self.n_observed >= self.maxlen:
            warnings.warn("history is full, overwriting old observations")
        index = self.n_observed % self.maxlen
        self.history[index, :self.input_d] = x
        self.history[index, self.input_d:] = y
        self.n_observed += 1


class BaseKNearestNeighborRegressor(NonParametricRegressor):
    def __init__(self, k=1, maxlen=1_000):
        super().__init__(maxlen=maxlen)
        self.k = k

    def predict(self, x):
        if self.history is None:
            return numpy.array([[numpy.nan]])
        distances = numpy.linalg.norm(self.history[:self.n_observed, :self.input_d] - numpy.squeeze(x), axis=1)
        try:
            k = min(self.k, self.n_observed)
            idx = numpy.argpartition(distances, k - 1)[:k]
        except ValueError:
            return numpy.nan * numpy.empty(shape=(self.output_d,))
        return self.history[idx, self.input_d:].mean(axis=0)


class BaseKernelRegressor(NonParametricRegressor):
    def __init__(self, length_scale=1, maxlen=100):
        # TODO: use the inverse of length_scale, it's unintuitive
        super().__init__(maxlen=maxlen)
        self.length_scale = length_scale

    def make_jax_pred_f(self):
        if self.history is None:
            def f(x):
                return numpy.array([[numpy.nan]])
        else:
            history = jnp.array(self.history)
            input_d = int(self.input_d)
            length_scale = float(self.length_scale)
            def f(x):
                distances = jnp.linalg.norm(history[:, :input_d] - jnp.squeeze(x), axis=1)
                distances = jnp.nan_to_num(distances, nan=jnp.inf)
                log_weights = -length_scale * distances ** 2
                log_sum = jax.scipy.special.logsumexp(log_weights)
                log_weights = log_weights - log_sum
                return jnp.exp(log_weights) @ history[:, input_d:]
        return f

    def predict(self, x):
        return numpy.array(self.make_jax_pred_f()(x))

    def cross_validate_length_scale(self, length_scales, depth=100, ratio=.9, n_train=None, rng=None):
        if rng is None:
            rng = numpy.random.default_rng()

        if self.n_observed < 10:
            raise Exception("not enough data")

        history = self.history[:self.n_observed]
        history = history[~numpy.isnan(history).any(axis=1)]

        n_total = history.shape[0]
        if n_train is None:
            n_train = int(ratio * n_total)
        else:
            assert ratio is None, "can't specify both n_train and ratio"

        test_reg: BaseKernelRegressor = copy.deepcopy(self)
        test_reg.n_observed = n_train

        errors = []
        error_stds = []
        for length_scale in length_scales:
            error = []
            for _ in range(depth):
                idx = rng.permutation(n_total)

                test_reg.length_scale = length_scale
                test_reg.history = history[idx[:n_train]]
                x = history[idx[n_train:], :self.input_d]
                y = history[idx[n_train:], self.input_d:]

                for inner_x, inner_y in zip(x, y):
                    error.append((inner_y - test_reg.predict(inner_x))**2)
            errors.append(numpy.mean(error))
            error_stds.append(numpy.std(error, ddof=1))
        errors = numpy.array(errors)
        error_stds = numpy.array(error_stds)
        return length_scales[numpy.argmin(errors + error_stds / numpy.sqrt(depth))], (length_scales, errors, error_stds)


class BaseMultiKernelRegressor:
    def __init__(self, length_scales=(1,1), maxlen=100):
        self.maxlen = maxlen
        self.input_histories = None
        self.output_history = None
        self.n_observed = 0

        self.length_scales = numpy.array(length_scales)

    def observe(self, x, y):
        if any([numpy.any(~numpy.isfinite(sub_x)) for sub_x in x]) or numpy.any(~numpy.isfinite(y)):
            warnings.warn("ignoring non-finite input")
            return

        if self.input_histories is None:
            self.input_histories = [numpy.zeros(shape=(self.maxlen, sub_x.size)) * numpy.nan for sub_x in x]
            self.output_history = numpy.zeros(shape=(self.maxlen, y.size))

        if self.n_observed == self.maxlen:
            warnings.warn("history is full, overwriting old observations")
        index = self.n_observed % self.maxlen

        for history, sub_x in zip(self.input_histories, x):
            history[index, :] = sub_x
        self.output_history[index, :] = y
        self.n_observed += 1


    def make_jax_pred_f(self):
        # TODO: precompute
        if self.input_histories is None:
            def f(x):
                return numpy.array([[numpy.nan]])
        else:
            input_histories = [jnp.array(h) for h in self.input_histories]
            output_history = jnp.array(self.output_history)
            def f(x, length_scales=jnp.array(self.length_scales)):
                distances = [-length_scale * jnp.linalg.norm(history - jnp.squeeze(sub_x), axis=1) ** 2 for
                             (sub_x, history, length_scale) in zip(x, input_histories, length_scales)]
                log_weights = jnp.array(distances).sum(axis=0)
                log_weights = jnp.nan_to_num(log_weights, nan=-numpy.inf)
                log_sum = jax.scipy.special.logsumexp(log_weights)
                log_weights = log_weights - log_sum

                return jnp.exp(log_weights) @ output_history
        return f

    def predict(self, x):
        # if self.input_histories is None:
        #     return numpy.array([[numpy.nan]])
        #
        # distances = [-length_scale*jnp.linalg.norm(history - jnp.squeeze(sub_x), axis=1)**2 for (sub_x, history, length_scale) in zip(x, self.input_histories, self.length_scales)]
        # log_weights = jnp.array(distances)
        # log_weights = jnp.nan_to_num(log_weights,nan=-numpy.inf)
        # log_sum = jax.scipy.special.logsumexp(log_weights, axis=1)
        # log_weights = log_weights - log_sum[:,None]
        # return numpy.array((self.kernel_weights@jnp.exp(log_weights)) @ self.output_history)

        return numpy.array(self.make_jax_pred_f()(x))



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
                super().observe(numpy.hstack([numpy.array(self._y_history).flatten(), x.flatten()]), y)

        def predict(self, x):
            if autoregress_only:
                x = 0 * x

            if len(self._y_history) == self._y_history.maxlen:
                return super().predict(numpy.hstack([numpy.array(self._y_history).flatten(), x.flatten()]))
            else:
                return numpy.array([[numpy.nan]])

    return AutoRegressor


class VanillaOnlineRegressor(DecoupledTransformer, BaseVanillaOnlineRegressor):
    base_algorithm = BaseVanillaOnlineRegressor

    def __init__(self, *, input_streams=None, output_streams=None, log_level=None, init_min_ratio=1.1, add_intercept=True, regularization_factor=0.01):
        input_streams = input_streams or {0: 'X', 1: 'Y'}
        DecoupledTransformer.__init__(self, input_streams=input_streams, output_streams=output_streams, log_level=log_level)
        BaseVanillaOnlineRegressor.__init__(self, init_min_ratio=init_min_ratio, regularization_factor=regularization_factor, add_intercept=add_intercept)
        self.log |= {'preq_error':[], 't': []}
        self.last_seen = {}

    def partial_fit(self, data, stream=0):
        if self.frozen:
            return
        self.pre_log_for_partial_fit(data, stream)
        self._partial_fit(data, stream)
        self.log_for_partial_fit(data, stream)

    def pre_log_for_partial_fit(self, data, stream):
        if self.log_level >= 2:
            stream_label = self.input_streams[stream]
            if stream_label in ('X', 'Y'):
                if numpy.isnan(data).any():
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
            if numpy.isnan(data).any():
                return

            self.last_seen[stream_label] = data
            if len(self.last_seen) == 2:
                for i in range(self.last_seen['X'].shape[0]):
                    self.observe(self.last_seen['X'][i], self.last_seen['Y'][i])
                self.last_seen = {}

    def transform(self, data, stream=0, return_output_stream=False):
        stream_label = self.input_streams[stream]
        if stream_label in {'X', 'qX'}:
            if numpy.isnan(data).any():
                data = numpy.nan * data
            else:
                prediction = [self.predict(row) for row in data]
                if isinstance(data, ArrayWithTime):
                    data = ArrayWithTime(prediction, data.t)
                else:
                    data = numpy.array(prediction)

        stream = self.output_streams[stream]

        return (data, stream) if return_output_stream else data

    def plot_preq_error(self, ax):
        t = numpy.array(self.log['t'])
        preq_error = numpy.array(self.log['preq_error'])
        sq_error = preq_error**2
        ax.plot(t, sq_error)
        ax.set_xlabel('time')
        ax.set_ylabel('regression training preqential error')
