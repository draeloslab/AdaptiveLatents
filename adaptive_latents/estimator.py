import contextlib
import copy
import pickle
import time
import typing
import warnings
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from frozendict import frozendict
from tqdm.auto import tqdm

from .timed_data_source import ArrayWithTime, GeneratorDataSource


class PassThroughDict(frozendict):
    def __missing__(self, key):
        return key

    def inverse_map(self, key):
        if key not in self.values():
            return key

        values = [k for k, v in self.items() if v == key]
        if len(values) == 0:
            raise IndexError('Key has no inverse.')
        elif len(values) > 1:
            raise IndexError('Key has too many inverses.')
        elif key not in self.keys():
            raise IndexError('Key has too many inverses (one of which is an implicit passthrough).')
        else:
            return values[0]


class StreamingEstimator(ABC):
    def __init__(self, input_streams=None, output_streams=None, log_level=None):
        """
        Parameters
        ----------
        input_streams: dict
            Keys are stream numbers, values are a flag to the transformer about how to process the data.
            So {3: 'X'} would mean that stream 3 should be processed as an X variable.
            Data not in an input_stream will usually be passed through.
        output_streams: dict[int, int]
            Keys are input streams, values are output streams; this is stream remapping applied after the transformer.
        log_level: int
            0: no logging
            1: profiling
            2: basic logging
            3: complete logging
        """

        self.input_streams = PassThroughDict(input_streams or {})
        self.output_streams = PassThroughDict(output_streams or {})
        self.log_level = log_level or 0
        self.mid_run_sources = None
        self.log = dict(step_time=[], stream=[])


    def step(self, data, stream=0, return_output_stream=False):
        """
        Learns and applies a transformation to incoming data.

        Parameters
        ----------
        data: any, np.ndarray
            data can be anything, but for most transformers it will be an array of shape (n_samples, sample_dimension)
        stream: int | typing.Hashable
            The stream the incoming data is coming from; 0 is the default.
            While this could technically be any hashable value, the convention is to use ints.
        return_output_stream: bool
            Whether to return the output stream; this is mostly only useful in pipelines, and so is false by default.

        Returns
        -------
        data
            the processed data
        stream: int, optional
            the stream the outputted data should be routed to
        """
        if self.log_level >= 1:
            start = time.time()
            self.log['stream'].append(stream)
            self.pre_log_for_step(data, stream)

        ret = self._step(data, stream, return_output_stream)

        if self.log_level >= 1:
            time_elapsed = time.time() - start
            if hasattr(data, 't'):
                time_elapsed = ArrayWithTime(time_elapsed, data.t)
            self.log['step_time'].append(time_elapsed)

            self.log_for_step(data, stream)
        return ret

    def pre_log_for_step(self, data, stream):
        pass

    def log_for_step(self, data, stream):
        pass


    @abstractmethod
    def _step(self, data, stream, return_output_stream):
        # most implementations will need to handle initialization and nan values; possibly also logging?
        stream = self.output_streams[stream]
        return (data, stream) if return_output_stream else data

    def blank_copy(self):
        return type(self)(**self.get_params())

    def trace_route(self, stream):
        middle_str = str(self) if stream in self.input_streams else ""
        if stream == self.output_streams[stream]:
            return middle_str
        return [stream, middle_str, self.output_streams[stream]]

    def _parse_sources(self, sources):
        if not (isinstance(sources, tuple) or isinstance(sources, list)):  # passed a single source
            sources = [sources]
        elif not len(sources): # passed an empty list
            warnings.warn('passed an empty sources list')
            return [], []

        if not isinstance(sources[0], tuple):  # passed a list of sources without streams
            streams = range(len(sources))
            sources = zip(sources, streams)

        sources, streams = zip(*sources)


        new_sources = []
        for source in sources:
            if isinstance(source, np.ndarray) and not isinstance(source, ArrayWithTime):
                source = ArrayWithTime.from_notime(source)
            elif not isinstance(source, np.ndarray):
                source = GeneratorDataSource(source)

            if isinstance(source, ArrayWithTime):
                source = copy.deepcopy(source)
                if len(source.shape) == 2:
                    source = source[:,None,:]
                    assert source.shape[0] == len(source.t)

            new_sources.append(source)
        sources = new_sources

        return sources, streams


    def streaming_run_on(self, sources, return_output_stream=False):
        """
        Parameters
        ----------
        sources: np.ndarray, types.GeneratorType, list[np.ndarray | types.GeneratorType], DataSource, list[DataSource], list[tuple[DataSource, int]], dict
            This should be the set of data sources.
            Inputs are parsed like this:
                a single array gets upgraded to a list: a -> [a]
                a list gets zipped with `range()`:  [a] -> [(a,0)]
                the elements returned from iter(a) will get fed into the 0 stream
        return_output_stream: bool
            Whether to yield the output stream or not. This is false by default to not confuse first-time users.

        Yields
        -------
        data: np.ndarray
            The processed version of each element of the given iterator.
        stream: int, optional
            the stream that the outputted data belongs to
        """

        sources, streams = self._parse_sources(sources)

        sources = list(zip(map(iter, sources), streams))
        self.mid_run_sources = sources
        while True:  # while-true/break is a code smell, but I want a do-while
            next_time = float('inf')
            for source, stream in reversed(sources):  # reversed to prefer the first element
                source_next_time = source.next_sample_time()
                if source_next_time <= next_time:
                    next_time = source_next_time
                    next_source, next_stream = source, stream
            if not next_time < float('inf'):
                break

            yield self.step(data=next(next_source), stream=next_stream, return_output_stream=return_output_stream)

        self.mid_run_sources = None

    def offline_run_on(self, sources, convinient_return=True, exit_time=None, show_tqdm=False):
        outputs = {}

        exit_time_for_tqdm = float('inf') if exit_time is None else exit_time

        pre_pbar = contextlib.nullcontext()
        if show_tqdm:
            for source in self._parse_sources(copy.deepcopy(sources))[0]:
                if hasattr(source, 't'):
                    exit_time_for_tqdm = min(exit_time_for_tqdm, source.t.max())
            pre_pbar = tqdm(total=None if exit_time_for_tqdm == float('inf') else round(exit_time_for_tqdm,2))

        with pre_pbar as pbar:
            for data, stream in self.streaming_run_on(sources, return_output_stream=True):
                if exit_time is not None and data.t > exit_time:
                    break
                if stream not in outputs:
                    outputs[stream] = []
                outputs[stream].append(data)
                if show_tqdm:
                    assert not isinstance(data.t, np.ndarray) or data.t.size == 1
                    pbar.update(round(float(data.t), 2) - pbar.n)

        if convinient_return:
            if isinstance(convinient_return, bool):
                convinient_return = 0

            if convinient_return not in outputs:
                warnings.warn(f"No outputs were routed to stream '{convinient_return}'.")
                outputs[convinient_return] = []

            data = outputs[convinient_return]
            outputs = ArrayWithTime.from_list(data, squeeze_type='to_2d', drop_early_nans=True)  # can be replaced with np.squeeze

        return outputs


    def __str__(self):
        kwargs = ', '.join(f'{k}={v}' for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({kwargs})"

    # for printing and testing
    def get_params(self, deep=True):
        # TODO: should this deep copy?
        return dict(input_streams=self.input_streams, output_streams=self.output_streams, log_level=self.log_level)

    # this is mostly for testing
    def expected_data_streams(self, rng, DIM, cycles=1):
        for _ in range(cycles):
            for s in self.input_streams:
                yield rng.normal(size=(10, DIM)), s

    @property
    def base_algorithm(self):
        """
        This is mostly for testing; it's useful for checking that e.g. ProSVD (the transformer) has the same arguments
        as BaseProSVD (which is not a transformer.)
        """
        return type(self)


class DecoupledEstimator(StreamingEstimator):
    def __init__(self, *, input_streams=None, output_streams=None, log_level=None):
        super().__init__(input_streams, output_streams, log_level)
        self.frozen = False

    def _step(self, data, stream=0, return_output_stream=False):
        self.partial_fit(data, stream)
        return self.transform(data, stream, return_output_stream)

    def partial_fit(self, data, stream=0) -> None:
        if self.frozen:
            return
        self._partial_fit(data, stream)

    @abstractmethod
    def _partial_fit(self, data, stream):
        """data should be of shape (n_samples, sample_size)"""
        # TODO: implement common functionality here
        pass

    @abstractmethod
    def transform(self, data, stream=0, return_output_stream=False):
        pass

    def freeze(self, b=True):
        self.frozen = b

    def offline_fit_then_transform(self, sources, convinient_return=True, exit_time=None):
        self.offline_run_on(sources, convinient_return, exit_time)
        self.freeze()
        return self.offline_run_on(sources, convinient_return, exit_time)

    def inverse_transform(self, data, stream=0, return_output_stream=False):
        raise NotImplementedError()



class Pipeline(DecoupledEstimator):
    def __init__(self, steps=(), *, input_streams=None, reroute_inputs=True, output_streams=None, log_level=None):
        self.steps: list[DecoupledEstimator] = steps
        self.reroute_inputs = reroute_inputs

        if input_streams is None:
            if reroute_inputs:
                expected_streams = set(k for step in self.steps for k in step.input_streams.keys())
                input_streams = dict(zip(range(len(expected_streams)), expected_streams))
            else:
                input_streams = PassThroughDict({})

        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)

    def get_params(self, deep=True):
        p = dict(steps=self.steps, reroute_inputs=self.reroute_inputs)
        if deep:
            for i, step in enumerate(self.steps):
                for k, v in step.get_params(deep).items():
                    p[f'__steps[{i}]__{k}'] = v
        return p | super().get_params(deep)

    def _partial_fit(self, data, stream=0):
        self.step(data, stream)

    def _step(self, data, stream=0, return_output_stream=False):
        stream = self.input_streams[stream]
        for step in self.steps:
            data, stream = step.step(data, stream=stream, return_output_stream=True)

        stream = self.output_streams[stream]
        if not return_output_stream:
            return data
        return data, stream

    def transform(self, data, stream=0, return_output_stream=False):
        stream = self.input_streams[stream]
        for step in self.steps:
            data, stream = step.transform(data, stream=stream, return_output_stream=True)
        stream = self.output_streams[stream]

        if not return_output_stream:
            return data
        return data, stream

    def inverse_transform(self, data, stream=0, return_output_stream=False):
        stream = self.output_streams.inverse_map(stream)
        for step in self.steps[::-1]:
            data, stream = step.inverse_transform(data, stream=stream, return_output_stream=True)
        stream = self.input_streams.inverse_map(stream)

        if not return_output_stream:
            return data

        return data, stream

    def freeze(self, b=True):
        self.frozen = b
        for step in self.steps:
            step.freeze(b)

    def trace_route(self, stream):
        super_path = [stream]

        path = []
        stream = self.input_streams[stream]
        for step in self.steps:
            path.append(step.trace_route(stream))
            stream = step.output_streams[stream]

        super_path.append(path)
        stream = self.output_streams[stream]
        super_path.append(stream)

        if super_path[0] == super_path[2]:
            return path
        return super_path

    def __str__(self):
        return f"{self.__class__.__name__}([{', '.join(str(s) for s in self.steps)}])"

class IgnoreDataEvent:
    def __init__(self, no_fit_interval, no_observe_interval=None, eps=0):
        if no_observe_interval is None:
            no_observe_interval = (np.inf, -np.inf)


        self.no_fit_interval = no_fit_interval
        self.no_observe_interval = no_observe_interval
        self.eps=eps

        for interval in [self.no_fit_interval, self.no_observe_interval]:
            if interval[1] < interval[0] and not (interval[0] == np.inf and interval[1] == -np.inf):
                raise ValueError()


    def get_data_observation_state(self, current_time) -> bool:
        return not (self.no_observe_interval[0] - self.eps <= current_time <= self.no_observe_interval[1] + self.eps)

    def get_parameter_fitting_state(self, current_time) -> bool:
        return not (self.no_fit_interval[0] - self.eps <= current_time <= self.no_fit_interval[1] + self.eps)

    def in_effect(self, current_time) -> bool:
        return min(self.no_fit_interval[0], self.no_observe_interval[0]) - self.eps <= current_time <= max(self.no_fit_interval[1], self.no_observe_interval[1]) + self.eps

    def has_passed(self, current_time) -> bool:
        return current_time > max(self.no_fit_interval[1], self.no_observe_interval[1]) + self.eps

    def __repr__(self):
        return f"{self.__class__.__name__}(no_fit_interval={self.no_fit_interval}, no_observe_interval={self.no_observe_interval})"



class Predictor(StreamingEstimator):
    stream_to_update_log_on = None
    def __init__(self, input_streams=None, output_streams=None, log_level=None, check_dt=False, n_steps_to_predict=1):
        input_streams = input_streams or {0: 'X', 1: 'dt_X'}
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)
        self.check_dt = check_dt
        self.dt = None
        self._last_X_t = None
        self._parameter_fitting_state = True
        self._data_observation_state = True
        self.ignore_data_events: None | list[IgnoreDataEvent] = None

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
        """returns nan if unitialized"""
        pass

    @abstractmethod
    def unevaluated_log_pred_p(self, n_steps):
        pass

    def get_data_observation_state(self):
        return self._data_observation_state
    def get_parameter_fitting_state(self):
        return self._parameter_fitting_state

    def set_parameter_fitting_state(self, value):
        assert self.ignore_data_events is None
        self._parameter_fitting_state = value
    def set_data_observation_state(self, value):
        assert self.ignore_data_events is None
        self._data_observation_state = value

    def update_states_based_on_events(self, current_time):
        if self.ignore_data_events is None:
            return

        current_events = [e for e in self.ignore_data_events if e.in_effect(current_time)]
        if len(current_events):
            if len(current_events) > 1:
                warnings.warn(f"there are currently {len(current_events)} overlapping events; this may cause unexpected behavior")
            # assert len(current_events) == 1, 'overlapping events are not currently supported'
            event = current_events[0]
            self._parameter_fitting_state = event.get_parameter_fitting_state(current_time)
            self._data_observation_state = event.get_data_observation_state(current_time)
        else:
            self._parameter_fitting_state = True
            self._data_observation_state = True

    def add_event(self, event):
        if self.ignore_data_events is None:
            self.ignore_data_events = []

        if isinstance(event, tuple):
            assert len(event) == 2
            assert isinstance(event[0], float) or isinstance(event[0], int)
            event = IgnoreDataEvent(no_fit_interval=event, no_observe_interval=None)

        self.ignore_data_events.append(event)

    def step(self, data, stream=0, return_output_stream=False):
        original_data = None
        if self.log_level >= 2:
            original_data = copy.deepcopy(data)

        if self.log_level >= 1:
            self.log['stream'].append(stream)

        start = time.time()
        ret = self._step(data, stream, return_output_stream)
        time_elapsed = time.time() - start

        if self.log_level >= 1:
            if hasattr(data, 't'):
                time_elapsed = ArrayWithTime(time_elapsed, data.t)
            self.log['step_time'].append(time_elapsed)

        self.log_for_step(data, stream, original_data=original_data)
        return ret

    def log_for_step(self, data, stream, original_data=None):
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



    def _step(self, data, stream, return_output_stream):
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
                if isinstance(self.dt, np.ndarray):
                    warnings.warn('dt is a numpy array; it is recommended that it is a hashable type')

            self.update_states_based_on_events(data.t)

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

        return (data, stream) if return_output_stream else data

    def data_to_n_steps(self, data):
        if isinstance(data, np.ndarray):
            assert data.size == 1
        q_dt = float(np.squeeze(data))
        if self.check_dt and self.dt is not None:
            steps = q_dt / self.dt
        else:
            steps = q_dt
        steps = float(steps)
        assert np.isclose(steps, steps := round(steps)), "without tracking dt, queries must be an integer number of steps"
        steps = int(steps)
        return steps

    def make_prediction_times(self, source, n_steps=1):
        dt = (source.dt if self.check_dt else 1) * n_steps
        return ArrayWithTime(np.ones_like(source.t).reshape(-1,1) * dt, source.t)

    @staticmethod
    def plot_pdf(fig, ax, pdf_f, xlim, ylim, native_d=3, e1=None, e2=None, density=100, add_colorbar=True):
        # TODO: move this to be a standalone in plotting_functions
        if e1 is None or e2 is None:
            assert e1 is None and e2 is None
            e1 = np.zeros(native_d)
            e2 = np.zeros(native_d)
            e1[0] = 1
            e2[1] = 1
        elif isinstance(e1,int):
            assert isinstance(e2,int)
            pre_e1 = np.zeros(native_d)
            pre_e2 = np.zeros(native_d)
            pre_e1[e1] = 1
            pre_e2[e2] = 1
            e1, e2 = pre_e1, pre_e2

        x_bins = np.linspace(*xlim, density + 1)
        y_bins = np.linspace(*ylim, density + 1)
        pdf_values = np.zeros(shape=(density, density))
        for i in range(density):
            for j in range(density):
                x = (x_bins[i] + x_bins[i + 1]) / 2
                y = (y_bins[j] + y_bins[j + 1]) / 2
                pdf_values[i, j] = pdf_f(x * e1 + y * e2)
        pdf_values = np.array(pdf_values)

        im = ax.pcolormesh(x_bins, y_bins, pdf_values.T, cmap='plasma')
        if add_colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

    def get_params(self, deep=True):
        return super().get_params(deep) | dict(check_dt=self.check_dt, n_steps_to_predict=self.n_steps_to_predict)


    def expected_data_streams(self, rng, DIM, cycles=1):
        dt = 1  # TODO: do this better
        start_t = self._last_X_t or -1
        for i in range(1, cycles+1):
            yield ArrayWithTime(rng.normal(size=(1, DIM)), t=i*dt + start_t), 'X'
            yield ArrayWithTime(np.ones((1, 1)) * dt, t=i*dt+ start_t), 'dt_X'



class TypicalEstimator(DecoupledEstimator):
    def __init__(self, *, input_streams=None, output_streams=None, log_level=None, on_nan_width=None):
        input_streams = input_streams or {0: 'X'}
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)
        self.is_initialized = False
        self.on_nan_width = on_nan_width

    def get_params(self, deep=True):
        p = super().get_params(deep)
        p = self.instance_get_params() | {'on_nan_width': self.on_nan_width} | p
        return p

    def _partial_fit(self, data, stream=0):
        if self.input_streams[stream] == 'X':
            if np.isnan(data).any():
                idx = np.isnan(data).any(axis=1)
                if idx.all():
                    return
                data = data[~np.isnan(data).any(axis=1)]

            if not self.is_initialized:
                self.pre_initialization_fit_for_X(data)
            else:
                self.partial_fit_for_X(data)

    def transform(self, data, stream=0, return_output_stream=False):
        if self.input_streams[stream] == 'X':
            if not self.is_initialized or np.isnan(data).any():
                if self.on_nan_width is None:
                    data = np.nan * data
                else:
                    data = (np.nan * data)[:,:self.on_nan_width]
            else:
                data = self.transform_for_X(data)

        stream = self.output_streams[stream]
        if return_output_stream:
            return data, stream
        return data

    def inverse_transform(self, data, stream=0, return_output_stream=False):
        stream = self.output_streams.inverse_map(stream)
        if self.input_streams[stream] == 'X':
            if not self.is_initialized or np.isnan(data).any():
                data = np.nan * data
            else:
                data = self.inverse_transform_for_X(data)

        if return_output_stream:
            return data, stream
        return data

    def pre_initialization_fit_for_X(self, X):
        self.is_initialized = True

    @abstractmethod
    def partial_fit_for_X(self, X):
        pass

    @abstractmethod
    def transform_for_X(self, X):
        pass

    @abstractmethod
    def instance_get_params(self, deep=True):
        pass

    def inverse_transform_for_X(self, X):
        raise NotImplementedError()


class CenteringEstimator(TypicalEstimator):
    def __init__(self, *, init_size=0, input_streams=None, output_streams=None, nan_when_uninitialized=False, on_nan_width=None, log_level=None):
        super().__init__(input_streams=input_streams, output_streams=output_streams, on_nan_width=on_nan_width, log_level=log_level)
        self.init_size = init_size
        self.samples_seen = 0
        self.center = 0
        self.nan_when_uninitialized = nan_when_uninitialized

    def add_new_input_channels(self, n):
        self.center = np.hstack([self.center, np.zeros(n)])
        self.samples_seen = np.hstack([self.samples_seen, np.zeros(n)])

    def pre_initialization_fit_for_X(self, X):
        self.partial_fit_for_X(X)
        # TODO: NaN for zero entries dynamically?
        if self.samples_seen.max() >= self.init_size:
            self.is_initialized = True

    def partial_fit_for_X(self, X):
        self.samples_seen += np.ones(X.shape[1])
        self.center = self.center + (X.sum(axis=0) - X.shape[0] * self.center) / self.samples_seen

    def transform_for_X(self, X):
        if not self.is_initialized and self.nan_when_uninitialized:
            return np.nan * X
        else:
            return X - self.center

    def inverse_transform_for_X(self, X):
        return X + self.center

    def instance_get_params(self, deep=True):
        return {'init_size': self.init_size, 'nan_when_uninitialized': self.nan_when_uninitialized}
         

class ZScoringEstimator(TypicalEstimator):
    # see https://math.stackexchange.com/a/1769248/701602
    """
    Examples
    --------
    >>> X = np.random.normal(size=(1000, 5)) * np.arange(5)
    >>> z = ZScoringEstimator(freeze_after_init=False)
    >>> _ = z.offline_run_on(X)
    >>> assert np.allclose(z.get_std(), np.std(X, axis=0), atol=0.01)
    """
    def __init__(self, *, init_size=10, freeze_after_init=False, input_streams=None, output_streams=None, on_nan_width=None, log_level=None):
        super().__init__(input_streams=input_streams, output_streams=output_streams, on_nan_width=on_nan_width, log_level=log_level)
        self.init_size = init_size
        self.freeze_after_init = freeze_after_init
        self.mean = 0
        self.m2 = 1e-8
        self.samples_seen = 0

    def pre_initialization_fit_for_X(self, X):
        self.partial_fit_for_X(X)
        if self.samples_seen >= self.init_size:
            self.is_initialized = True
            if self.freeze_after_init:
                self.freeze(True)

    def partial_fit_for_X(self, X):
        for x in X:
            delta = x - self.mean
            self.mean += delta / (self.samples_seen + 1)
            self.m2 += delta * (x - self.mean)
            self.samples_seen += 1

    def transform_for_X(self, X):
        return (X - self.mean) / self.get_std()

    def get_std(self):
        return np.sqrt(self.m2 / (self.samples_seen - 1))

    def instance_get_params(self, deep=True):
        return dict(init_size=self.init_size, freeze_after_init=self.freeze_after_init)


class KernelSmoother(StreamingEstimator):
    def __init__(self, *, tau=1, kernel_length=None, custom_kernel=None, input_streams=None, output_streams=None, log_level=None):
        input_streams = input_streams or {0:'X'}
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)
        self.tau = tau
        self.kernel_length = kernel_length
        self.custom_kernel = custom_kernel
        if custom_kernel is None:
            delta_t = 1 # todo: make time-aware
            alpha = 1 - np.exp(-delta_t/tau)
            if kernel_length is None:
                kernel_length = np.ceil(tau * 5).astype(int)

            kernel = alpha * (1-alpha)**np.arange(kernel_length)[::-1]
        else:
            kernel = custom_kernel
        self.kernel = kernel
        self.last_X = None
        self.history = deque(maxlen=len(self.kernel))

    def _step(self, data, stream, return_output_stream):
        if self.input_streams[stream] == 'X':
            output = []
            for row in data:
                self.history.append(row)
                if len(self.history) >= len(self.kernel) and not np.isnan(a:=np.array(self.history)).any():
                    output.append(self.kernel @ a)
                else:
                    output.append(np.nan*row)
            data = ArrayWithTime.from_transformed_data(output, data)
        stream = self.output_streams[stream]
        return (data, stream) if return_output_stream else data

    def add_new_input_channels(self, n):
        for i in range(len(self.history)):
            self.history[i] = np.hstack((self.history[i], np.zeros(n)))

    def get_params(self, deep=True):
        return dict(tau=self.tau, kernel_length=self.kernel_length, custom_kernel=self.custom_kernel) | super().get_params()

    def plot_impulse_response(self, ax):
        """
        Parameters
        ----------
        ax: matplotlib.pyplot.Axes
            The axis to plot on.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> KernelSmoother().plot_impulse_response(ax)
        """

        impulse_point = len(self.kernel) + 3
        a = np.zeros((2*impulse_point + 1,1,1))
        a[impulse_point] = 1
        b = np.array(self.offline_run_on(a, convinient_return=False)[0])
        ax.plot(a[:,0,0], '.-', label='original signal')
        ax.plot(b[:,0,0], '.-', label='smoothed signal')
        ax.axvline(len(self.kernel), color='k', linestyle='--', label='end of initialization')
        # ax.axvline(impulse_point + len(self.kernel), color='k', alpha=.25)
        # ax.axvline(impulse_point, color='k', alpha=.25, label='region of impulse response')
        ax.fill_between([impulse_point, impulse_point + len(self.kernel)-1], 1,  color='k', alpha=.1, label='impulse response')
        ax.legend()



class Concatenator(StreamingEstimator):

    def __init__(self, *, input_streams=None, output_streams=None, log_level=None, stream_scaling_factors=None):
        input_streams = input_streams or PassThroughDict({0:0, 1:1})

        output_stream = max(input_streams.keys()) + 1
        output_streams = output_streams or PassThroughDict({k: output_stream for k in input_streams.keys()} | {'skip': -1})
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)
        self.last_seen = {}

        if stream_scaling_factors is None:
            stream_scaling_factors = {i:1 for i in self.input_streams}

        self.stream_scaling_factors = stream_scaling_factors

    def _step(self, data, stream, return_output_stream):
        if stream in self.input_streams:
            self.last_seen[self.input_streams[stream]] = data

            if len(self.last_seen) == len(self.input_streams):
                data = [(k, v) for k, v in self.last_seen.items()]
                data.sort()
                data = [(k, v * self.stream_scaling_factors[k] if k in self.stream_scaling_factors else v) for k, v in data]
                data = np.hstack([v for k, v in data])
                if all([isinstance(x, ArrayWithTime) for x in self.last_seen.values()]):
                    t = max((x.t for x in self.last_seen.values()))
                    data = ArrayWithTime(input_array=data, t=t)
                self.last_seen = {}
            else:
                data = np.nan * data
                stream = 'skip'

        stream = self.output_streams[stream]
        return data, stream if return_output_stream else data

    def get_params(self, deep=True):
        p = dict(stream_scaling_factors=self.stream_scaling_factors)
        return p | super().get_params(deep)


class Tee(DecoupledEstimator):
    def __init__(self, input_streams=None, log_level=None, output_streams=None):
        input_streams = input_streams or PassThroughDict()
        self.observed = {}
        super().__init__(input_streams=input_streams, log_level=log_level, output_streams=output_streams)

    def _partial_fit(self, data, stream):
        if stream in self.input_streams:
            semantic_stream = self.input_streams[stream]
            if semantic_stream not in self.observed:
                self.observed[semantic_stream] = []
            self.observed[semantic_stream].append(data)

    def transform(self, data, stream=0, return_output_stream=False):
        return (data, stream) if return_output_stream else data

    def convert_to_array(self):
        self.observed = {k: ArrayWithTime.from_list(v, squeeze_type='to_2d', drop_early_nans=True) for k, v in self.observed.items()}
        return self.observed
