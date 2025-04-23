import warnings
from collections import deque

import numpy as np

from . import StreamingKalmanFilter
from .predictor import Predictor
from .regressions import BaseKNearestNeighborRegressor, OnlineRegressor
from .timed_data_source import ArrayWithTime
from .transformer import StreamingTransformer


# class StimRegressorOld(StreamingTransformer):
#     def __init__(self, autoreg=None, stim_reg=None, attempt_correction=True, input_streams=None, output_streams=None, log_level=None, ):
#         input_streams = input_streams or {0: 'stim', 1: 'X', 2: 'dt_X'}
#         super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)
#         if autoreg is None:
#             autoreg = StreamingKalmanFilter()
#         self.autoreg: Predictor = autoreg
#         self.attempt_correction = attempt_correction
#         if stim_reg is None:
#             stim_reg = BaseKNearestNeighborRegressor(k=2)
#         self.stim_reg: OnlineRegressor = stim_reg
#         self.last_seen_stims = deque(maxlen=1)
#
#     def _partial_fit_transform(self, data, stream, return_output_stream):
#         if self.input_streams[stream] == 'X':
#             data_depth = 1
#             assert data.shape[0] == data_depth, data.shape
#
#             if np.isfinite(data).all():
#                 if self.last_seen_stims and np.any(self.last_seen_stims[-1]):
#                     pred = self.autoreg.predict(n_steps=1)
#                     residual = data - pred
#
#                     stim_reg_input = np.hstack([self.autoreg.predict(n_steps=0).flatten(), self.last_seen_stims[-1].flatten()])
#                     self.stim_reg.observe(stim_reg_input, residual)
#
#                     self.autoreg.toggle_parameter_fitting(False)
#                     self.autoreg.observe(data, stream=self.input_streams[stream])
#                     self.autoreg.toggle_parameter_fitting(True)
#                 else:
#                     self.autoreg.observe(data, stream=self.input_streams[stream])
#             else:
#                 warnings.warn('there should probably be an autonomous dynamics call here')
#
#             data = ArrayWithTime.from_transformed_data(self.autoreg.get_prediction_state().reshape(data_depth, -1), data)
#
#         elif self.input_streams[stream] == 'dt_X':
#             steps = self.autoreg.data_to_n_steps(data)
#             pred = self.autoreg.predict(n_steps=steps)
#
#             if np.isfinite(pred).all():
#                 if self.last_seen_stims and np.any(self.last_seen_stims[-1]) and self.attempt_correction:
#                     stim_reg_input = np.hstack([self.autoreg.predict(n_steps=0).flatten(), self.last_seen_stims[-1].flatten()])
#                     pred = pred + self.stim_reg.predict(stim_reg_input)
#
#             data = ArrayWithTime.from_transformed_data(pred.reshape(1,-1), data)
#
#         elif self.input_streams[stream] == 'stim':
#             self.last_seen_stims.append(data)
#
#         return (data, stream) if return_output_stream else data
#
#     def get_params(self, deep=True):
#         return super().get_params(deep) | dict(autoreg=self.autoreg, stim_reg=self.stim_reg,
#                                                attempt_correction=self.attempt_correction)
#
#     # this is mostly for testing
#     def expected_data_streams(self, rng, DIM):
#         # TODO: do this better
#         return [
#             (rng.normal(size=(1, DIM)), 'X'),
#             (np.ones((1, 1)), 'dt_X'),
#             (np.zeros((1, 1)) * (rng.random() > .9), 'toggle_parameter_fitting'),
#         ]

class StimRegressor(Predictor):
    def __init__(self, autoreg=None, stim_reg=None, attempt_correction=True, n_steps_to_predict=1, check_dt=None, input_streams=None, output_streams=None, log_level=None, ):
        input_streams = input_streams or {0: 'stim', 1: 'X', 2: 'dt_X'}
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level, check_dt=check_dt, n_steps_to_predict=n_steps_to_predict)
        if autoreg is None:
            autoreg = StreamingKalmanFilter()
        self.autoreg: Predictor = autoreg
        self.attempt_correction = attempt_correction
        if stim_reg is None:
            stim_reg = BaseKNearestNeighborRegressor(k=2)
        self.stim_reg: OnlineRegressor = stim_reg
        self.last_seen_stims = deque(maxlen=1)

    def predict(self, n_steps):
        assert n_steps in {0,1}

        pred = self.autoreg.predict(n_steps=n_steps)

        if n_steps == 1:
            pred = self.autoreg.predict(n_steps=n_steps)

            if self.last_seen_stims and np.any(self.last_seen_stims[-1]) and self.attempt_correction and np.isfinite(pred).all():
                    stim_reg_input = np.hstack([self.autoreg.predict(n_steps=0).flatten(), self.last_seen_stims[-1].flatten()])
                    pred = pred + self.stim_reg.predict(stim_reg_input)

        return pred

    def unevaluated_log_pred_p(self, n_steps):
        assert n_steps in {0,1}

        if self.last_seen_stims and np.any(self.last_seen_stims[-1]) and self.attempt_correction:
            stim_reg_input = np.hstack([self.autoreg.predict(n_steps=0).flatten(), self.last_seen_stims[-1].flatten()])
            offset = self.stim_reg.predict(stim_reg_input)

        inner_f = self.autoreg.unevaluated_log_pred_p(n_steps=n_steps)
        def f(future_point):
            return inner_f(future_point - offset)
        return f

    def observe(self, X, stream=None):
        if self.last_seen_stims and np.any(self.last_seen_stims[-1]):
            pred = self.autoreg.predict(n_steps=1)
            residual = X - pred

            stim_reg_input = np.hstack([self.autoreg.predict(n_steps=0).flatten(), self.last_seen_stims[-1].flatten()])
            self.stim_reg.observe(stim_reg_input, residual)

            self.autoreg.toggle_parameter_fitting(False)
            self.autoreg.observe(X, stream=self.input_streams[stream])
            self.autoreg.toggle_parameter_fitting(True)
        else:
            self.autoreg.observe(X, stream=self.input_streams[stream])

    def get_prediction_state(self):
        return self.autoreg.get_prediction_state()

    def get_arbitrary_dynamics_parameter(self):
        return self.autoreg.get_arbitrary_dynamics_parameter()

    def _partial_fit_transform(self, data, stream, return_output_stream):
        if self.input_streams[stream] == 'stim':
            self.last_seen_stims.append(data)
        else:
            data, stream = super()._partial_fit_transform(data, stream, return_output_stream=True)

        return (data, stream) if return_output_stream else data

    def toggle_parameter_fitting(self, value=None):
        self.autoreg.toggle_parameter_fitting(value)

    def get_params(self, deep=True):
        return super().get_params(deep) | dict(autoreg=self.autoreg, stim_reg=self.stim_reg, attempt_correction=self.attempt_correction)