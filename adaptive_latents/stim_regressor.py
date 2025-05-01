from collections import deque

import numpy as np

from . import StreamingKalmanFilter
from .predictor import Predictor
from .regressions import BaseKNearestNeighborRegressor, OnlineRegressor
from .timed_data_source import ArrayWithTime
from .stim_optimization import StimDesigner


class StimRegressor(Predictor):
    stream_to_log_on = 'stim'
    def __init__(self, autoreg=None, stim_reg=None, stim_designer=None, heed_stimuli=True, attempt_correction=True, input_streams=None, output_streams=None, log_level=None, check_dt=True, n_steps_to_predict=1):
        input_streams = input_streams or {0: 'stim', 1: 'X', 2: 'dt_X'}
        assert n_steps_to_predict == 1
        assert heed_stimuli or not attempt_correction  # correcting without learning doesn't make sense
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level, check_dt=check_dt, n_steps_to_predict=n_steps_to_predict)
        if autoreg is None:
            autoreg = StreamingKalmanFilter()
        if stim_designer is None:
            stim_designer = StimDesigner()
        self.stim_designer = stim_designer
        self.autoreg: Predictor = autoreg
        self.attempt_correction = attempt_correction
        self.heed_stimuli = heed_stimuli
        if stim_reg is None:
            stim_reg = BaseKNearestNeighborRegressor(k=2)
        self.stim_reg: OnlineRegressor = stim_reg
        self.last_seen_stims = deque(maxlen=1)
        self.s_hat_error_function = None # TODO: delete this, it's a hack

    def _partial_fit_transform(self, data, stream, return_output_stream):
        if self.input_streams[stream] == 'stim':
            self.last_seen_stims.append(data)
            ret =  (data, stream) if return_output_stream else data
        else:
            ret = super()._partial_fit_transform(data, stream, return_output_stream)

        return ret

    def log_for_partial_fit(self, data, stream, original_data=None):
        super().log_for_partial_fit(data, stream, original_data=original_data)
        if self.s_hat_error_function is not None and self.input_streams[stream] == 'X' and self.last_seen_stims and np.any(self.last_seen_stims[-1]):
            key = 's_hat_error'
            if key not in self.log:
                self.log[key] = []
            self.log[key].append(ArrayWithTime.from_transformed_data(self.s_hat_error_function(self), data))


    def predict(self, n_steps):
        assert n_steps in {0,1}
        pred = self.autoreg.predict(n_steps=n_steps)

        if np.isfinite(pred).all():
            if self.last_seen_stims and np.any(self.last_seen_stims[-1]) and self.attempt_correction:
                pred = pred + self.predict_stim_response()

        return pred

    def predict_stim_response(self):
        stim_reg_input = np.hstack([self.autoreg.predict(n_steps=0).flatten(), self.last_seen_stims[-1].flatten()])
        return self.stim_reg.predict(stim_reg_input)

    def observe(self, X, stream=None):
        if self.last_seen_stims and np.any(self.last_seen_stims[-1]) and self.heed_stimuli:
            pred = self.autoreg.predict(n_steps=1)
            residual = X - pred

            stim_reg_input = np.hstack([self.autoreg.predict(n_steps=0).flatten(), self.last_seen_stims[-1].flatten()])
            self.stim_reg.observe(stim_reg_input, residual)

            self.autoreg.toggle_parameter_fitting(False)
            self.autoreg.observe(X, stream=self.input_streams[stream])
            self.autoreg.toggle_parameter_fitting(True)
        else:
            self.autoreg.observe(X, stream=self.input_streams[stream])

    def get_state(self):
        return self.autoreg.get_state()

    def get_arbitrary_dynamics_parameter(self):
        return self.autoreg.get_arbitrary_dynamics_parameter()

    def unevaluated_log_pred_p(self, n_steps):
        assert n_steps in {0,1}
        f = self.autoreg.unevaluated_log_pred_p(n_steps=n_steps)

        if self.last_seen_stims and np.any(self.last_seen_stims[-1]) and self.attempt_correction:
            correction = self.predict_stim_response()
            def corrected_f(future_point):
                return f(future_point - correction)
        else:
            corrected_f = f
        return corrected_f

    def get_params(self, deep=True):
        return super().get_params(deep) | dict(autoreg=self.autoreg, stim_reg=self.stim_reg, attempt_correction=self.attempt_correction, heed_stimuli=self.heed_stimuli)

