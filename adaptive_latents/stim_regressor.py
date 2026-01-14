import functools
from collections import deque

import numpy as np

from .input_sources.kalman_filter import StreamingKalmanFilter
from .estimator import Predictor
from .regressions import BaseKNearestNeighborRegressor, OnlineRegressor, BaseMultiKernelRegressor
from .timed_data_source import ArrayWithTime

# TODO: make the time comparisons more uniform

dt_epsilon = 1e-8

class StimAutoReg():
    def __init__(self, n_steps_to_consider):
        self.n_steps_to_consider = n_steps_to_consider
        self.previous_corrections = []
        self.training_data = []
        self.coeffs = np.zeros(n_steps_to_consider) * np.nan

    def correct(self, current_t, dt):
        new_correction = 0
        for correction in reversed(self.previous_corrections):
            steps = (current_t - correction.t) / dt
            # assert abs(steps - round(steps)) < dt/4, steps # TODO: is this ok to ignore?
            steps = int(round(steps))

            if steps >= self.n_steps_to_consider:
                break
            new_correction += correction * self.coeffs[steps-1]
        return new_correction

    def observe_new_correction(self, new_correction):
        self.previous_corrections.append(np.squeeze(new_correction))
        self.training_data.append([])

    def observe(self, X, pred_callback, dt):
        if len(self.previous_corrections) == 0:
            return

        steps = (X.t - self.previous_corrections[-1].t)/dt
        if steps >= self.n_steps_to_consider + 2: # TODO: simplify this logic
            return
        steps = int(round(steps))
        if not abs(steps - round(steps)) < dt/5:  # TODO: make this standard
            print(f'{steps=} {X.t=}')
            raise Exception()
        if steps >= self.n_steps_to_consider + 1:
            return

        pred = pred_callback()
        residual = X - pred
        self.training_data[-1].append(np.squeeze(residual))

        if len(self.training_data) > 1 and type(self.training_data[-2]) is list:
            if len(self.training_data[-2]) == self.n_steps_to_consider:
                self.training_data[-2] = self.training_data[-2]
            else:
                self.training_data.pop(-2)
            errors = np.array(self.training_data[:-1])
            corrections = np.array(self.previous_corrections[:-1])[:,None,:]

            corrections = corrections.transpose((0,2,1))
            errors = errors.transpose((0,2,1))
            self.coeffs, _, _, _ = np.linalg.lstsq(corrections.reshape((-1, 1)), errors.reshape((-1, self.n_steps_to_consider)))
            self.coeffs = self.coeffs.flatten()

class StimRegressor(Predictor):
    stream_to_update_log_on = 'stim'
    def __init__(self, autoreg=None, stim_reg=None, heed_stimuli=True, attempt_correction=True, error_on_missed_stim=True, input_streams=None, output_streams=None, log_level=None, check_dt=True, n_steps_to_predict=1, stim_delay=0):
        input_streams = input_streams or {0: 'X', 1: 'stim', 2: 'dt_X'}
        assert n_steps_to_predict == 1
        assert heed_stimuli or not attempt_correction  # correcting without learning doesn't make sense
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level, check_dt=check_dt, n_steps_to_predict=n_steps_to_predict)

        if autoreg is None:
            autoreg = StreamingKalmanFilter()
        self.autoreg: Predictor = autoreg
        if stim_reg is None:
            stim_reg = BaseMultiKernelRegressor(maxlen=100)
        self.stim_reg: BaseMultiKernelRegressor = stim_reg
        self.attempt_correction = attempt_correction
        self.heed_stimuli = heed_stimuli
        self.last_seen_stims = deque()
        self.stim_autoreg = StimAutoReg(n_steps_to_consider=0)
        assert stim_delay >= 0
        self.stim_delay = stim_delay  # in units of time (wrt the data)
        self.error_on_missed_stim = error_on_missed_stim

    def _step(self, data, stream, return_output_stream):
        if self.input_streams[stream] == 'stim':
            if self.is_notable_stim(data):
                self.last_seen_stims.append(data)
            ret =  (data, stream) if return_output_stream else data
        else:
            ret = super()._step(data, stream, return_output_stream)

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
        saftey_margin = self.dt*1.2 if self.dt else self.stim_delay # TODO: check this timing/synchronization logic
        while self.last_seen_stims and (current_t - self.last_seen_stims[0].t) > (self.stim_delay + saftey_margin):
            if self.error_on_missed_stim and self.heed_stimuli and np.isfinite(self.autoreg.get_arbitrary_dynamics_parameter()).all():
                raise MissedStimulusError(f"Missed stim. {current_t=:.3f} {self.last_seen_stims[0].t=:.3f} (diff={current_t-self.last_seen_stims[0].t:.2f}) {(self.stim_delay + saftey_margin)=:.3f}")
            self.last_seen_stims.popleft()

    def get_stim_to_correct_for(self, current_t, remove=False):
        to_return = []
        for stim in self.last_seen_stims:
            if np.isclose(stim.t + self.stim_delay, current_t, atol=self.dt/20):
                to_return.append(stim)

        if remove:
            for stim in to_return:
                self.last_seen_stims.remove(stim)

        if len(to_return) == 0:
            return []
        elif len(to_return) == 1:
            return to_return[0].flatten()
        else:
            raise Exception("Can only correct for one stimulus at a time.")



    def log_for_step(self, data, stream, original_data=None):
        super().log_for_step(data, stream, original_data=original_data)

        if self.log_level >= 2 and self.dt is not None:
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


    def predict_stim_response(self, stim_to_correct_for, current_t):
        # TODO: is current_t correct here?
        stim_reg_input = [self.autoreg.predict(n_steps=0).flatten(), stim_to_correct_for, current_t]
        return self.stim_reg.predict(stim_reg_input)

    def observe(self, X, stream=None):
        if self.heed_stimuli and self.in_stim_lag(current_t=X.t):
            self.autoreg.toggle_parameter_fitting(False)

            stim_to_correct_for = self.get_stim_to_correct_for(current_t=X.t, remove=True)
            if len(stim_to_correct_for):
                self.autoreg.toggle_parameter_fitting(False)
                pred = self.autoreg.predict(n_steps=1)
                residual = X - pred
                stim_reg_input = [self.autoreg.predict(n_steps=0).flatten(), stim_to_correct_for, X.t]  # TODO: deal with nan from autoreg
                self.stim_reg.observe(stim_reg_input, residual)
                self.stim_autoreg.observe_new_correction(ArrayWithTime(self.stim_reg.predict(stim_reg_input), X.t))

            # TODO: make a decision about wheither autoreg needs to be a transformer
            # self.autoreg.observe(X, stream=self.input_streams[stream])
            self.autoreg.step(data=X, stream=self.input_streams[stream])
        else:
            self.autoreg.toggle_parameter_fitting(True)
            self.stim_autoreg.observe(X,functools.partial(self.autoreg.predict,n_steps=1), self.dt)
            self.autoreg.step(data=X, stream=self.input_streams[stream])

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
            current_t = current_t + self.dt * n_steps
            stim_to_correct_for = self.get_stim_to_correct_for(current_t=current_t)
            if len(stim_to_correct_for):
                pred = pred + self.predict_stim_response(stim_to_correct_for, current_t)
            pred = pred + self.stim_autoreg.correct(current_t, self.dt)
        return pred

    def unevaluated_log_pred_p(self, n_steps, current_t=None):
        if current_t is None:
            current_t = self._last_X_t
        assert n_steps in {0,1}
        f = self.autoreg.unevaluated_log_pred_p(n_steps=n_steps)

        if self.attempt_correction:
            current_t = self.dt * n_steps + current_t
            stim_to_correct_for = self.get_stim_to_correct_for(current_t=current_t)
            if len(stim_to_correct_for):
                correction = self.predict_stim_response(stim_to_correct_for, current_t)
            else:
                correction = 0
            def corrected_f(future_point):
                return f(future_point - correction)
        else:
            corrected_f = f
        return corrected_f

    def finalize_log(self, stim_intended_samples=None):
        self.log['pred_error'] = ArrayWithTime.from_list(self.log['pred_error'], drop_early_nans=True, squeeze_type='to_2d')
        if stim_intended_samples is not None:
            self.log['stim_intended_samples'] = stim_intended_samples.slice((stim_intended_samples > 0).any(axis=1))

    def get_params(self, deep=True):
        return super().get_params(deep) | dict(autoreg=self.autoreg, stim_reg=self.stim_reg, attempt_correction=self.attempt_correction, heed_stimuli=self.heed_stimuli, stim_delay=self.stim_delay, error_on_missed_stim=self.error_on_missed_stim)

    def __getstate__(self):
        # TODO: check for jax?
        self.unevaluated_log_pred_ps = {}
        return super().__getstate__()


class MissedStimulusError(RuntimeError):
    pass