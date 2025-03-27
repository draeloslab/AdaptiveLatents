from collections import deque

import numpy as np

from . import StreamingKalmanFilter
from .predictor import Predictor
from .regressions import BaseKNearestNeighborRegressor, OnlineRegressor
from .timed_data_source import ArrayWithTime
from .transformer import StreamingTransformer


class StimRegressor(StreamingTransformer):
    def __init__(self, autoreg=None, stim_reg=None, attempt_correction=True, input_streams=None, output_streams=None,
                 log_level=None, ):
        input_streams = input_streams or {0: 'stim', 1: 'X', 2: 'dt_X'}
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)
        if autoreg is None:
            autoreg = StreamingKalmanFilter()
        self.autoreg: Predictor = autoreg
        self.attempt_correction = attempt_correction
        if stim_reg is None:
            stim_reg = BaseKNearestNeighborRegressor(k=2)
        self.stim_reg: OnlineRegressor = stim_reg
        self.last_seen_stims = deque(maxlen=1)

    def _partial_fit_transform(self, data, stream, return_output_stream):
        if self.input_streams[stream] == 'X':
            data_depth = 1
            assert data.shape[0] == data_depth, data.shape

            if np.isfinite(data).all():
                if self.last_seen_stims and self.last_seen_stims[-1]:
                    pred = self.autoreg.predict(n_steps=1)
                    residual = data - pred
                    self.stim_reg.observe(pred, residual)

                    self.autoreg.toggle_parameter_fitting(False)
                    self.autoreg.observe(data, stream=self.input_streams[stream])
                    self.autoreg.toggle_parameter_fitting(True)
                else:
                    self.autoreg.observe(data, stream=self.input_streams[stream])

            data = ArrayWithTime.from_transformed_data(self.autoreg.get_state().reshape(data_depth, -1), data)

        elif self.input_streams[stream] == 'dt_X':
            steps = self.autoreg.data_to_n_steps(data)
            pred = self.autoreg.predict(n_steps=steps)

            if np.isfinite(pred).all():
                if self.last_seen_stims and self.last_seen_stims[-1] and self.attempt_correction:
                    pred = pred + self.stim_reg.predict(pred)

            data = ArrayWithTime.from_transformed_data(pred, data)

        elif self.input_streams[stream] == 'stim':
            self.last_seen_stims.append(data)

        return (data, stream) if return_output_stream else data

    def get_params(self, deep=True):
        return super().get_params(deep) | dict(autoreg=self.autoreg, stim_reg=self.stim_reg,
                                               attempt_correction=self.attempt_correction)

    # this is mostly for testing
    def expected_data_streams(self, rng, DIM):
        # TODO: do this better
        return [
            (rng.normal(size=(1, DIM)), 'X'),
            (np.ones((1, 1)), 'dt_X'),
            (np.zeros((1, 1)) * (rng.random() > .9), 'toggle_parameter_fitting'),
        ]
