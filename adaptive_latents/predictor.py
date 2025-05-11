import copy
from abc import abstractmethod
import time
import warnings

import numpy as np
import pytest

from .timed_data_source import ArrayWithTime
from .transformer import StreamingTransformer


class Predictor(StreamingTransformer):
    stream_to_update_log_on = None
    def __init__(self, input_streams=None, output_streams=None, log_level=None, check_dt=False, n_steps_to_predict=1):
        input_streams = input_streams or {0: 'X', 1: 'dt_X', 'toggle_parameter_fitting': 'toggle_parameter_fitting'}
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)
        self.check_dt = check_dt
        self.dt = None
        self._last_X_t = None
        self.parameter_fitting = True

        self.n_steps_to_predict = n_steps_to_predict
        self.unevaluated_log_pred_ps = {}
        self.predictions = {}

    @abstractmethod
    def predict(self, n_steps):
        pass

    @abstractmethod
    def observe(self, X, stream=None):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_arbitrary_dynamics_parameter(self):
        pass

    @abstractmethod
    def unevaluated_log_pred_p(self, n_steps):
        pass


    def partial_fit_transform(self, data, stream=0, return_output_stream=False):
        original_data = None
        if self.log_level >= 2:
            original_data = copy.deepcopy(data)

        if self.log_level >= 1:
            self.log['stream'].append(stream)

        start = time.time()
        ret = self._partial_fit_transform(data, stream, return_output_stream)
        time_elapsed = time.time() - start

        if self.log_level >= 1:
            if hasattr(data, 't'):
                time_elapsed = ArrayWithTime(time_elapsed, data.t)
            self.log['step_time'].append(time_elapsed)

        self.log_for_partial_fit(data, stream, original_data=original_data)
        return ret

    def log_for_partial_fit(self, data, stream, original_data=None):
        if self.log_level >= 2:
            assert self.check_dt
            if 'pred_error' not in self.log:
                for k in ['pred_error', 'log_pred_p', 'log_pred_p_origin_t', 'pred_origin_t']:
                    self.log[k] = []

            if self.dt is not None:
                if self.input_streams[stream] == 'X':
                    current_t = data.t
                    real_time_offset = self.dt * self.n_steps_to_predict

                    # normal error calculation
                    for t_to_eval in list(self.predictions.keys()):
                        if np.isclose(t_to_eval - current_t, 0, atol=self.dt/10):
                            origin_t, prediction = self.predictions[t_to_eval]
                            self.log['pred_error'].append(ArrayWithTime(prediction - original_data, current_t))
                            self.log['pred_origin_t'].append(origin_t)
                            del self.predictions[t_to_eval]
                        elif t_to_eval < current_t:
                            del self.predictions[t_to_eval]

                    # log pred p calculation
                    for t_to_eval in list(self.unevaluated_log_pred_ps.keys()):
                        if np.isclose(t_to_eval - current_t, 0, atol=self.dt/10):
                            origin_t, pdf = self.unevaluated_log_pred_ps[t_to_eval]
                            self.log['log_pred_p'].append(ArrayWithTime(pdf(original_data), current_t))
                            self.log['log_pred_p_origin_t'].append(origin_t)
                            del self.unevaluated_log_pred_ps[t_to_eval]
                        elif t_to_eval < current_t:
                            del self.unevaluated_log_pred_ps[t_to_eval]

                    self.predictions[current_t + real_time_offset] = (current_t, self.predict(self.n_steps_to_predict))
                    self.unevaluated_log_pred_ps[current_t + real_time_offset] = (current_t, self.unevaluated_log_pred_p(self.n_steps_to_predict))


    def toggle_parameter_fitting(self, value=None):
        if value is not None:
            self.parameter_fitting = bool(value)
        else:
            self.parameter_fitting = not self.parameter_fitting

    def _partial_fit_transform(self, data, stream, return_output_stream):
        if self.input_streams[stream] == 'X':
            if self.check_dt:
                assert hasattr(data, 't')
                if self._last_X_t is not None:
                    dt = data.t - self._last_X_t
                    assert dt > 0
                    if self.dt is not None:
                        consistent_dt = np.isclose(data.t - self._last_X_t, self.dt)
                        # assert consistent_dt, 'time steps for training are not consistent'
                        if not consistent_dt:
                            warnings.warn('time steps for training are not consistent')
                        self.dt = (self.dt + dt)/2
                    else:
                        self.dt = dt
                self._last_X_t = data.t

            data_depth = 1
            assert data.shape[0] == data_depth

            if np.isfinite(data).all():
                self.observe(data, stream=stream)
            else:
                warnings.warn('there should probably be an autonomous dynamics call here')

            data = ArrayWithTime.from_transformed_data(self.get_state().reshape(data_depth,-1), data)

        elif self.input_streams[stream] == 'dt_X':
            steps = self.data_to_n_steps(data)
            pred = self.predict(n_steps=steps)
            data = ArrayWithTime.from_transformed_data(pred, data)
        elif self.input_streams[stream] == 'toggle_parameter_fitting':
            self.toggle_parameter_fitting(data)

        return (data, stream) if return_output_stream else data

    def data_to_n_steps(self, data):
        assert data.size == 1
        q_dt = data[0, 0]
        if self.check_dt and self.dt is not None:
            steps = q_dt / self.dt
        else:
            steps = q_dt

        assert np.isclose(steps, steps := round(steps)), "without tracking dt, queries must be an integer number of steps"
        steps = int(steps)
        return steps

    def make_prediction_times(self, source, n_steps=1):
        dt = (source.dt if self.check_dt else 1) * n_steps
        return ArrayWithTime(np.ones_like(source.t).reshape(-1,1) * dt, source.t)

    @staticmethod
    def plot_pdf(fig, ax, pdf_f, xlim, ylim, native_d=3, e1=None, e2=None, density=100):
        # TODO: move this to be a standalone in plotting_functions
        if e1 is None or e2 is None:
            assert e1 is None and e2 is None
            e1 = np.zeros(native_d)
            e2 = np.zeros(native_d)
            e1[0] = 1
            e2[1] = 1

        x_bins = np.linspace(*xlim, density + 1)
        y_bins = np.linspace(*ylim, density + 1)
        pdf_values = np.zeros(shape=(density, density))
        for i in range(density):
            for j in range(density):
                x = (x_bins[i] + x_bins[i + 1]) / 2
                y = (y_bins[j] + y_bins[j + 1]) / 2
                pdf_values[i, j] = pdf_f(x * e1 + y * e2)
        pdf_values = np.array(pdf_values)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax.pcolormesh(x_bins, y_bins, pdf_values.T, cmap='plasma')
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.axis('equal')

    def get_params(self, deep=True):
        return super().get_params(deep) | dict(check_dt=self.check_dt, n_steps_to_predict=self.n_steps_to_predict)


    def expected_data_streams(self, rng, DIM, cycles=1):
        dt = 1  # TODO: do this better
        start_t = self._last_X_t or -1
        for i in range(1, cycles+1):
            yield ArrayWithTime(rng.normal(size=(1, DIM)), t=i*dt + start_t), 'X'
            yield ArrayWithTime(np.ones((1, 1)) * dt, t=i*dt+ start_t), 'dt_X'
            yield ArrayWithTime(np.ones((1, 1)) * (rng.random() > .9), t=i*dt+ start_t), 'toggle_parameter_fitting'

    @classmethod
    def test_if_api_compatible(cls, constructor=None, rng=None, DIM=None):
        constructor, rng, DIM = super().test_if_api_compatible(constructor, rng, DIM)
        cls._test_checks_dt(constructor, rng, DIM)
        cls._test_output_t_is_origin_t(constructor, rng, DIM)

    @staticmethod
    def _test_output_t_is_origin_t(constructor, rng, DIM):
        predictor: Predictor = constructor()

        predictor.offline_run_on(rng.normal(size=(100, DIM)))

        output = predictor.partial_fit_transform(ArrayWithTime([[1]], t=100), stream='dt_X')
        assert np.all(output.t == 100)

    @staticmethod
    def _test_checks_dt(constructor, rng, DIM):

        predictor: Predictor = constructor(check_dt=True)
        dt = 1/np.pi
        predictor.partial_fit_transform(ArrayWithTime(rng.normal(size=(1, DIM)), 0), stream='X')
        predictor.partial_fit_transform(ArrayWithTime(rng.normal(size=(1, DIM)), 1 * dt), stream='X')

        assert np.isclose(predictor.dt, dt)

        predictor_backup = copy.deepcopy(predictor)

        # pytest_condition = pytest.raises(AssertionError)
        pytest_condition = pytest.warns(UserWarning, match='time steps for training are not consistent')

        with pytest_condition:
            warnings.warn('time steps for training are not consistent')


        predictor = copy.deepcopy(predictor_backup)
        with pytest.raises(AssertionError):
            predictor.partial_fit_transform(ArrayWithTime(rng.normal(size=(1, DIM)), 1 * dt), stream='X')

        predictor = copy.deepcopy(predictor_backup)
        with pytest_condition:
            predictor.partial_fit_transform(ArrayWithTime(rng.normal(size=(1, DIM)), 3 * dt), stream='X')

        predictor = copy.deepcopy(predictor_backup)
        predictor.partial_fit_transform(ArrayWithTime(rng.normal(size=(1, DIM)), 2 * dt), stream='X')

        predictor = copy.deepcopy(predictor_backup)
        with pytest.raises(AssertionError):
            predictor.partial_fit_transform(ArrayWithTime([[1]], 3 * dt), stream='dt_X')
