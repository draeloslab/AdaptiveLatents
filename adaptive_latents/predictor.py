from abc import abstractmethod
import numpy as np
from .transformer import StreamingTransformer
from .timed_data_source import ArrayWithTime
import copy


class Predictor(StreamingTransformer):
    def __init__(self, input_streams=None, output_streams=None, log_level=None, check_dt=False):
        input_streams = input_streams or {0: 'X', 1: 'dt_X', 'toggle_parameter_fitting': 'toggle_parameter_fitting'}
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)
        self.check_dt = check_dt
        self.dt = None
        self._last_t = None
        self.parameter_fitting = True

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

    def toggle_parameter_fitting(self, value=None):
        if value is not None:
            self.parameter_fitting = bool(value)
        else:
            self.parameter_fitting = not self.parameter_fitting

    def _partial_fit_transform(self, data, stream, return_output_stream):
        if self.input_streams[stream] == 'X':
            if self.check_dt:
                assert hasattr(data, 't')
                if self._last_t is not None:
                    dt = data.t - self._last_t
                    assert dt > 0
                    if self.dt is not None:
                        assert np.isclose(data.t - self._last_t, self.dt), 'time steps for training are not consistent'
                        self.dt = (self.dt + dt)/2
                    else:
                        self.dt = dt
                self._last_t = data.t

            assert data.shape[0] == 1

            self.observe(data, stream=stream)

            data = ArrayWithTime.from_transformed_data(self.get_state(), data)

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

    def get_params(self, deep=True):
        return super().get_params(deep) | dict(check_dt=self.check_dt)

    # this is mostly for testing
    def expected_data_streams(self, rng, DIM):
        # TODO: do this better
        return [
            (rng.normal(size=(1, DIM)), 'X'),
            (np.ones((1,1)), 'dt_X'),
            (np.zeros((1,1)) * (rng.random() > .9), 'toggle_parameter_fitting'),
        ]

    @classmethod
    def test_if_api_compatible(cls, constructor=None, rng=None, DIM=None):
        constructor, rng, dim = super().test_if_api_compatible(constructor, rng, DIM)
        cls._test_checks_dt(constructor, rng, DIM)

    @staticmethod
    def _test_checks_dt(constructor, rng, DIM):
        import pytest

        predictor: Predictor = constructor(check_dt=True)
        dt = 1/np.pi
        predictor.partial_fit_transform(ArrayWithTime(rng.normal(size=(1, DIM)), 0), stream='X')
        predictor.partial_fit_transform(ArrayWithTime(rng.normal(size=(1, DIM)), 1 * dt), stream='X')

        assert np.isclose(predictor.dt, dt)

        predictor_backup = copy.deepcopy(predictor)

        predictor = copy.deepcopy(predictor_backup)
        with pytest.raises(AssertionError):
            predictor.partial_fit_transform(ArrayWithTime(rng.normal(size=(1, DIM)), 1 * dt), stream='X')

        predictor = copy.deepcopy(predictor_backup)
        with pytest.raises(AssertionError):
            predictor.partial_fit_transform(ArrayWithTime(rng.normal(size=(1, DIM)), 3 * dt), stream='X')

        predictor = copy.deepcopy(predictor_backup)
        predictor.partial_fit_transform(ArrayWithTime(rng.normal(size=(1, DIM)), 2 * dt), stream='X')

        predictor = copy.deepcopy(predictor_backup)
        with pytest.raises(AssertionError):
            predictor.partial_fit_transform(ArrayWithTime([[1]], 3 * dt), stream='dt_X')
