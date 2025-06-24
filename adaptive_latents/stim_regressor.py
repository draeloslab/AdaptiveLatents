from collections import deque

import numpy as np

from . import StreamingKalmanFilter
from .predictor import Predictor
from .regressions import BaseKNearestNeighborRegressor, OnlineRegressor, BaseKernelRegressor
from .timed_data_source import ArrayWithTime
from .stim_designer import StimDesigner


class StimRegressor(Predictor):
    stream_to_update_log_on = 'stim'
    def __init__(self, autoreg=None, stim_reg=None, stim_designer=None, heed_stimuli=True, attempt_correction=True, input_streams=None, output_streams=None, log_level=None, check_dt=True, n_steps_to_predict=1, stim_delay=0):
        input_streams = input_streams or {0: 'stim', 1: 'X', 2: 'dt_X'}
        assert n_steps_to_predict == 1
        assert heed_stimuli or not attempt_correction  # correcting without learning doesn't make sense
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level, check_dt=check_dt, n_steps_to_predict=n_steps_to_predict)

        if autoreg is None:
            autoreg = StreamingKalmanFilter()
        self.autoreg: Predictor = autoreg
        if stim_designer is None:
            stim_designer = StimDesigner()
        self.stim_designer = stim_designer
        if stim_reg is None:
            # stim_reg = BaseKNearestNeighborRegressor(k=2)
            stim_reg = BaseKernelRegressor()
        self.stim_reg: BaseKernelRegressor = stim_reg
        self.attempt_correction = attempt_correction
        self.heed_stimuli = heed_stimuli
        self.last_seen_stims = deque()
        assert stim_delay >= 0
        self.stim_delay = stim_delay  # in units of time (wrt the data)
        self.s_hat_error_function = None # TODO: delete this, it's a hack

    def _partial_fit_transform(self, data, stream, return_output_stream):
        if self.input_streams[stream] == 'stim':
            if self.is_notable_stim(data):
                self.last_seen_stims.append(data)
            ret =  (data, stream) if return_output_stream else data
        else:
            ret = super()._partial_fit_transform(data, stream, return_output_stream)

        if hasattr(data, 't'):
            self.trim_last_seen_stims(current_t=data.t)

        return ret

    @staticmethod
    def is_notable_stim(stim):
        return (stim!=0).any()

    def in_stim_lag(self, current_t):
        for stim in self.last_seen_stims:
            dt = self.dt
            if dt is None:
                dt = self.stim_delay # TODO: is this a good idea?
            if stim.t + self.stim_delay + dt/10 >= current_t and self.is_notable_stim(stim):
                return True
        return False

    def trim_last_seen_stims(self, current_t):
        saftey_margin = self.dt if self.dt else self.stim_delay
        while self.last_seen_stims and (current_t - self.last_seen_stims[0].t) > (self.stim_delay + saftey_margin):
            self.last_seen_stims.popleft()

    def get_stim_to_correct_for(self, current_t):
        to_return = []
        for stim in self.last_seen_stims:
            if np.isclose(stim.t + self.stim_delay, current_t, atol=self.dt/20):
                to_return.append(stim)

        assert len(to_return) < 2
        if len(to_return) == 0:
            return []
        elif len(to_return) == 1:
            return to_return[0].flatten()



    def log_for_partial_fit(self, data, stream, original_data=None):
        super().log_for_partial_fit(data, stream, original_data=original_data)

        if self.log_level >= 2 and self.dt is not None:
            if self.input_streams[stream] == 'X' and len(self.get_stim_to_correct_for(data.t)) and self.s_hat_error_function is not None:
                key = 's_hat_error'
                if key not in self.log:
                    self.log[key] = []
                self.log[key].append(ArrayWithTime.from_transformed_data(self.s_hat_error_function(self), data))

            if self.input_streams[stream] == 'stim':
                real_time_offset = self.dt * self.n_steps_to_predict
                assert self.n_steps_to_predict == 1
                current_t_as_of_last_x = self._last_X_t
                prediction_time = current_t_as_of_last_x + real_time_offset
                for saved_prediction_time in self.predictions.keys():
                    if np.isclose(current_t_as_of_last_x - saved_prediction_time, current_t_as_of_last_x - prediction_time, rtol=.05):
                        prediction_time = saved_prediction_time

                self.predictions[prediction_time] = (current_t_as_of_last_x, self.predict(self.n_steps_to_predict))
                self.unevaluated_log_pred_ps[prediction_time] = (current_t_as_of_last_x, self.unevaluated_log_pred_p(self.n_steps_to_predict))


    def predict_stim_response(self, stim_to_correct_for):
        stim_reg_input = np.hstack([self.autoreg.predict(n_steps=0).flatten(), stim_to_correct_for])
        return self.stim_reg.predict(stim_reg_input)

    def observe(self, X, stream=None):
        if self.heed_stimuli and self.in_stim_lag(current_t=X.t):
            self.autoreg.toggle_parameter_fitting(False)

            stim_to_correct_for = self.get_stim_to_correct_for(current_t=X.t)
            if len(stim_to_correct_for):
                self.autoreg.toggle_parameter_fitting(False)
                pred = self.autoreg.predict(n_steps=1)
                residual = X - pred
                stim_reg_input = np.hstack([self.autoreg.predict(n_steps=0).flatten(), stim_to_correct_for])
                self.stim_reg.observe(stim_reg_input, residual)

            # TODO: make a decision about wheither autoreg needs to be a transformer
            # self.autoreg.observe(X, stream=self.input_streams[stream])
            self.autoreg.partial_fit_transform(data=X, stream=self.input_streams[stream])
        else:
            self.autoreg.toggle_parameter_fitting(True)
            self.autoreg.partial_fit_transform(data=X, stream=self.input_streams[stream])

    def get_state(self):
        return self.autoreg.get_state()

    def get_arbitrary_dynamics_parameter(self):
        return self.autoreg.get_arbitrary_dynamics_parameter()

    def predict(self, n_steps, current_t=None):
        if current_t is None:
            current_t = self._last_X_t
        assert n_steps in {0,1}
        pred = self.autoreg.predict(n_steps=n_steps)

        if self.attempt_correction and np.isfinite(pred).all():
            stim_to_correct_for = self.get_stim_to_correct_for(current_t=current_t + self.dt * n_steps)
            if len(stim_to_correct_for):
                pred = pred + self.predict_stim_response(stim_to_correct_for)
        return pred

    def unevaluated_log_pred_p(self, n_steps, current_t=None):
        if current_t is None:
            current_t = self._last_X_t
        assert n_steps in {0,1}
        f = self.autoreg.unevaluated_log_pred_p(n_steps=n_steps)

        if self.attempt_correction:
            stim_to_correct_for = self.get_stim_to_correct_for(current_t=self.dt * n_steps+current_t)
            if len(stim_to_correct_for):
                correction = self.predict_stim_response(stim_to_correct_for)
            else:
                correction = 0
            def corrected_f(future_point):
                return f(future_point - correction)
        else:
            corrected_f = f
        return corrected_f

    def get_params(self, deep=True):
        return super().get_params(deep) | dict(autoreg=self.autoreg, stim_reg=self.stim_reg, attempt_correction=self.attempt_correction, heed_stimuli=self.heed_stimuli, stim_designer=self.stim_designer, stim_delay=self.stim_delay)

    def __getstate__(self):
        # TODO: check for jax?
        self.unevaluated_log_pred_ps = {}
        return super().__getstate__()
