import functools
import warnings
from collections import deque

import numpy as np
from numba.core.ir import Raise

from .input_sources.kalman_filter import StreamingKalmanFilter
from .estimator import Predictor, IgnoreDataEvent
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

class StimEvent(IgnoreDataEvent):
    def __init__(self, no_fit_interval, difference_interval, u, no_observe_interval=None, delivery_time=None, eps=0, error_on_missed=True):
        super().__init__(no_fit_interval=no_fit_interval, no_observe_interval=no_observe_interval, eps=eps)

        self.delivery_time = delivery_time
        self.difference_interval = difference_interval
        self.u = u
        self.state_at_pred = None
        self.fufilled = False
        self.error_on_missed = error_on_missed
        self.predictions = {}

        if difference_interval[1] < difference_interval[0]:
            raise ValueError()

    def in_effect(self, current_time) -> bool:
        return min(self.no_fit_interval[0], self.no_observe_interval[0], self.difference_interval[0]) - self.eps <= current_time <= max(self.no_fit_interval[1], self.no_observe_interval[1], self.difference_interval[1]) + self.eps

    def has_passed(self, current_time) -> bool:
        has_passed = current_time > max(self.no_fit_interval[1], self.no_observe_interval[1], self.difference_interval[1]) + self.eps
        if has_passed and not self.fufilled and self.error_on_missed:
            raise MissedStimulusError()
        return has_passed

    def time_to_predict(self, current_time) -> bool:
        return self.difference_interval[0] - self.eps <= current_time <= self.difference_interval[0] + self.eps

    def time_to_correct(self, current_time) -> bool:
        return self.difference_interval[1] - self.eps <= current_time <= self.difference_interval[1] + self.eps

    def get_prediction_for_time(self, current_time):
        to_return = []
        for k, v in self.predictions.items():
            if k - self.eps <= current_time <= k + self.eps:
                to_return.append(v)
        assert len(to_return) > 0, 'missed prediction?'
        assert len(to_return) < 2, 'multiple predictions for the same time'
        return to_return[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(no_fit_interval={self.no_fit_interval}, no_observe_interval={self.no_observe_interval}, difference_interval={self.difference_interval})"


class StimRegressor(Predictor):
    stream_to_update_log_on = 'stim'
    def __init__(self, autoreg=None, stim_reg=None, heed_stimuli=True, attempt_correction=True, error_on_missed_stim=True, input_streams=None, output_streams=None, log_level=None, check_dt=True, n_steps_to_predict=1, stim_delay=0):
        input_streams = input_streams or {0: 'X', 1: 'stim', 2: 'dt_X'}
        assert n_steps_to_predict == 1
        assert heed_stimuli or not attempt_correction, "correcting without learning doesn't make sense"
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level, check_dt=check_dt, n_steps_to_predict=n_steps_to_predict)

        if autoreg is None:
            autoreg = StreamingKalmanFilter()
        self.autoreg: Predictor = autoreg
        if stim_reg is None:
            stim_reg = BaseMultiKernelRegressor(maxlen=100)
        self.stim_reg: BaseMultiKernelRegressor = stim_reg
        self.attempt_correction = attempt_correction
        self.heed_stimuli = heed_stimuli
        self.ignore_data_events: list[StimEvent] = []
        self.stim_autoreg = StimAutoReg(n_steps_to_consider=0)
        assert stim_delay >= 0
        self.stim_delay = stim_delay  # in units of time (wrt the data)
        self.error_on_missed_stim = error_on_missed_stim

    def _step(self, data, stream, return_output_stream):
        if self.input_streams[stream] == 'stim':
            if self.is_notable_stim_u(data):
                self.add_event(StimEvent(
                    no_fit_interval=(data.t, data.t + self.stim_delay),
                    no_observe_interval=None,
                    difference_interval=(data.t + self.stim_delay - self.dt, data.t + self.stim_delay),
                    delivery_time=data.t,
                    u=data,
                    error_on_missed=self.error_on_missed_stim,
                    eps=self.dt/8
                ))
            ret =  (data, stream) if return_output_stream else data
        else:
            ret = super()._step(data, stream, return_output_stream)

        return ret

    @staticmethod
    def is_notable_stim_u(stim):
        return (stim!=0).any()

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
        # from the present
        stim_reg_input = [self.autoreg.predict(n_steps=0).flatten(), stim_to_correct_for, current_t]
        return self.stim_reg.predict(stim_reg_input)

    def update_states_based_on_events(self, current_time):
        if not self.heed_stimuli:
            return
        super().update_states_based_on_events(current_time)
        self.autoreg.set_parameter_fitting_state(self._parameter_fitting_state)
        self.autoreg.set_data_observation_state(self._data_observation_state)

    def get_stim_to_predict_for(self, current_t) -> StimEvent | None:
        if self.ignore_data_events is None:
            return
        hits = []
        for stim in self.ignore_data_events:
            if hasattr(stim, 'time_to_predict') and stim.time_to_predict(current_t):
                hits.append(stim)

        if len(hits) > 1:
            raise ValueError()
        elif len(hits) == 1:
            return hits[0]
        else:
            return None

    def get_stim_to_correct_for(self, current_t) -> StimEvent | None:
        if self.ignore_data_events is None:
            return
        hits = []
        for stim in self.ignore_data_events:
            if hasattr(stim, 'time_to_correct') and stim.time_to_correct(current_t):
                hits.append(stim)

        if len(hits) > 1:
            raise ValueError()
        elif len(hits) == 1:
            return hits[0]
        else:
            return None

    def observe(self, X, stream=None):
        if self.dt is not None:
            stim_to_predict_for = self.get_stim_to_predict_for(current_t=X.t-self.dt)
            if self.heed_stimuli and stim_to_predict_for is not None:
                n_steps = self.data_to_n_steps(stim_to_predict_for.difference_interval[1] - stim_to_predict_for.difference_interval[0])
                assert n_steps == 1
                pred = self.autoreg.predict(n_steps=n_steps)
                stim_to_predict_for.predictions[X.t - self.dt + n_steps * self.dt] = pred
                stim_to_predict_for.state_at_pred = self.autoreg.predict(n_steps=0).flatten()


        stim_to_correct_for = self.get_stim_to_correct_for(current_t=X.t)
        if self.heed_stimuli and stim_to_correct_for is not None:
            stim_to_correct_for: StimEvent

            pred = stim_to_correct_for.get_prediction_for_time(X.t)
            stim_to_correct_for.fufilled = True
            state_at_pred = stim_to_correct_for.state_at_pred
            u = stim_to_correct_for.u
            delivery_time = stim_to_correct_for.delivery_time

            residual = X - pred
            stim_reg_input = [state_at_pred, u, np.array(delivery_time)]  # TODO: deal with nan from autoreg
            self.stim_reg.observe(stim_reg_input, residual)
            self.stim_autoreg.observe_new_correction(ArrayWithTime(self.stim_reg.predict(stim_reg_input), X.t))

        else:
            self.stim_autoreg.observe(X,functools.partial(self.autoreg.predict,n_steps=1), self.dt) # TODO: why is there a partial here?

        self.autoreg.step(data=X, stream=self.input_streams[stream])



    def get_state(self):
        return self.autoreg.get_state()

    def get_arbitrary_dynamics_parameter(self):
        return self.autoreg.get_arbitrary_dynamics_parameter()

    def predict(self, n_steps, current_t=None):
        if current_t is None:
            current_t = self._last_X_t
        if n_steps not in {0,1}:
            warnings.warn("predicting ahead more than 1 step with StimRegressor isn't officially supported, and may give inaccurate results")
        pred = self.autoreg.predict(n_steps=n_steps)

        if self.attempt_correction and np.isfinite(pred).all():
            current_t = current_t + self.dt * n_steps
            stim_to_correct_for = self.get_stim_to_correct_for(current_t=current_t)
            if stim_to_correct_for is not None:
                pred = pred + self.predict_stim_response(stim_to_correct_for.u, current_t)
            pred = pred + self.stim_autoreg.correct(current_t, self.dt)
        return pred

    def unevaluated_log_pred_p(self, n_steps, current_t=None):
        if current_t is None:
            current_t = self._last_X_t
        if n_steps not in {0,1}:
            warnings.warn("predicting ahead more than 1 step with StimRegressor isn't officially supported, and may give inaccurate results")
        f = self.autoreg.unevaluated_log_pred_p(n_steps=n_steps)

        if self.attempt_correction:
            current_t = self.dt * n_steps + current_t
            stim_to_correct_for = self.get_stim_to_correct_for(current_t=current_t)
            if stim_to_correct_for is not None:
                correction = self.predict_stim_response(stim_to_correct_for.u, current_t)
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

    def add_event(self, event):
        if self.ignore_data_events is None:
            self.ignore_data_events = []

        self.ignore_data_events.append(event)


class MissedStimulusError(RuntimeError):
    pass