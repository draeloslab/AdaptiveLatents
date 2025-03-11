import contextlib
import copy
import pickle
import time
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


class StreamingTransformer(ABC):
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

    def partial_fit_transform(self, data, stream=0, return_output_stream=False):
        """
        Learns and applies a transformation to incoming data.

        Parameters
        ----------
        data: any, np.ndarray
            data can be anything, but for most transformers it will be an array of shape (n_samples, sample_dimension)
        stream: int
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
        start = time.time()
        if self.log_level >= 1:
            self.log['stream'].append(stream)
        ret = self._partial_fit_transform(data, stream, return_output_stream)
        time_elapsed = time.time() - start

        if self.log_level >= 1:
            if hasattr(data, 't'):
                time_elapsed = ArrayWithTime(time_elapsed, data.t)
            self.log['step_time'].append(time_elapsed)

        self.log_for_partial_fit(data, stream)
        return ret

    def log_for_partial_fit(self, data, stream):
        pass

    @abstractmethod
    def _partial_fit_transform(self, data, stream, return_output_stream):
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

            yield self.partial_fit_transform(data=next(next_source), stream=next_stream, return_output_stream=return_output_stream)

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
            if 0 not in outputs:
                raise Exception("No outputs were routed to stream 0.")

            data = outputs[0]
            while data and np.isnan(data[0]).any():
                data.pop(0)
            outputs = ArrayWithTime.from_list(data, squeeze_type='to_2d')  # can be replaced with np.squeeze

        return outputs


    def __str__(self):
        kwargs = ', '.join(f'{k}={v}' for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({kwargs})"

    # for printing and testing
    def get_params(self, deep=True):
        # TODO: should this deep copy?
        return dict(input_streams=self.input_streams, output_streams=self.output_streams, log_level=self.log_level)

    # this is mostly for testing
    def expected_data_streams(self, rng, DIM):
        for s in self.input_streams:
            yield rng.normal(size=(10, DIM)), s

    @property
    def base_algorithm(self):
        """
        This is mostly for testing; it's useful for checking that e.g. ProSVD (the transformer) has the same arguments
        as BaseProSVD (which is not a transformer.)
        """
        return type(self)


    @classmethod
    def test_if_api_compatible(cls, constructor=None, rng=None, DIM=None):
        constructor = constructor or cls
        rng = rng or np.random.default_rng()
        DIM = DIM or 6

        cls._test_get_params_works(constructor)
        cls._test_can_fit_transform(constructor, rng, DIM)

        import pathlib
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            cls._test_can_save_and_rerun(constructor, rng, tmp_path=pathlib.Path(tmp_dir), DIM=DIM)

        return constructor, rng, DIM

    @staticmethod
    def _test_can_fit_transform(constructor, rng, DIM=6):
        transformer: StreamingTransformer = constructor()
        for _ in range(5):
            for data, s in transformer.expected_data_streams(rng, DIM):
                transformer.partial_fit_transform(data, s)

    @staticmethod
    def _test_can_save_and_rerun(constructor, rng, tmp_path, DIM=6):
        transformer: StreamingTransformer = constructor()

        for _ in range(5):
            for data, s in transformer.expected_data_streams(rng, DIM):
                transformer.partial_fit_transform(data, s)
        t2 = copy.deepcopy(transformer)

        temp_file = tmp_path / 'streaming_transformer.pickle'
        with open(temp_file, 'bw') as f:
            pickle.dump(transformer, f)

        del transformer

        with open(temp_file, 'br') as f:
            transformer = pickle.load(f)

        for data, s in transformer.expected_data_streams(rng, DIM):
            assert np.array_equal(transformer.partial_fit_transform(data, s), t2.partial_fit_transform(data, s),
                                  equal_nan=True)

    @staticmethod
    def _test_get_params_works(constructor):
        import inspect
        transformer: StreamingTransformer = constructor()
        p = {k: v for k, v in transformer.get_params().items() if len(k) and k[0] != "_"}
        type(transformer)(**p)

        base_algorithm = transformer.base_algorithm
        base_args = set(inspect.signature(base_algorithm).parameters.keys()) - {'args', 'kwargs'}
        found_args = set(p.keys()) - {'args', 'kwargs'}
        assert base_args.issubset(found_args)



class DecoupledTransformer(StreamingTransformer):
    def __init__(self, *, input_streams=None, output_streams=None, log_level=None):
        super().__init__(input_streams, output_streams, log_level)
        self.frozen = False

    def _partial_fit_transform(self, data, stream=0, return_output_stream=False):
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

    @classmethod
    def test_if_api_compatible(cls, constructor=None, rng=None, DIM=None):
        constructor, rng, DIM = super().test_if_api_compatible(constructor=constructor, rng=rng, DIM=DIM)

        cls._test_can_ignore_nans(constructor, rng)
        cls._test_original_matrix_unchanged(constructor, rng)
        cls._test_partial_fit_transform_decomposes_correctly(constructor, rng, DIM=DIM)
        cls._test_freezing_works_correctly(constructor, rng)
        cls._test_inverse_transform_works(constructor, rng, DIM=DIM)

        return constructor, rng, DIM

    @staticmethod
    def _make_sources(transformer, rng, expression=None, first_n_nan=0, length=20, DIM=6):
        import itertools
        if expression is None:
            expression = lambda: rng.normal(size=(3, DIM))

        batches = [expression() * (np.nan if i < first_n_nan else 1) for i in range(length)]
        return [tuple(x) for x in zip(itertools.repeat(batches), transformer.input_streams.keys())]

    @classmethod
    def _test_can_ignore_nans(cls, constructor, rng):
        transformer = constructor()

        sources = cls._make_sources(transformer, rng, first_n_nan=7)
        transformer.offline_run_on(sources, convinient_return=False)

        sources = cls._make_sources(transformer, rng)
        output = transformer.offline_run_on(sources, convinient_return=False)

        for stream in output:
            assert (~np.isnan(output[stream][-1])).all()

    @classmethod
    def _test_original_matrix_unchanged(cls, constructor, rng):
        transformer: DecoupledTransformer = constructor()

        sources = cls._make_sources(transformer, rng)
        transformer.offline_run_on(sources, convinient_return=False)

        for f in (transformer.partial_fit, transformer.transform):
            A = rng.normal(size=(1, 6))
            A_original = A.copy()
            f(A)
            assert np.all(A == A_original)

    @staticmethod
    def _test_partial_fit_transform_decomposes_correctly(constructor, rng, DIM=6):
        transformer: DecoupledTransformer = constructor()

        for i in range(20):
            for stream in transformer.input_streams.keys():
                batch = rng.normal(size=(3, DIM))

                t1 = transformer
                t2 = copy.deepcopy(transformer)

                o1 = t1.partial_fit_transform(batch, stream)

                t2.partial_fit(batch, stream)
                o2 = t2.transform(batch, stream)

                assert np.array_equal(o1, o2, equal_nan=True)

    @staticmethod
    def _test_freezing_works_correctly(constructor, rng):
        transformer: DecoupledTransformer = constructor()

        transformer.freeze(False)
        for i in range(10):
            for stream in transformer.input_streams.keys():
                batch = rng.normal(size=(2, 6))
                transformer.partial_fit(batch, stream)
        t2 = copy.deepcopy(transformer)

        transformer.freeze(True)
        for i in range(10):
            for stream in transformer.input_streams.keys():
                batch = rng.normal(size=(2, 6))
                transformer.partial_fit(batch, stream)
                assert np.array_equal(transformer.transform(batch), t2.transform(batch))

        transformer.freeze(False)
        for i in range(10):
            for stream in transformer.input_streams.keys():
                batch = rng.normal(size=(2, 6))
                transformer.partial_fit(batch, stream)
                t2.partial_fit(batch, stream)

                assert np.array_equal(transformer.transform(batch), t2.transform(batch))

    @classmethod
    def _test_inverse_transform_works(cls, constructor, rng, DIM=6):
        transformer: DecoupledTransformer = constructor()

        sources = cls._make_sources(transformer, rng)
        transformer.offline_run_on(sources, convinient_return=False)
        try:
            output = transformer.inverse_transform(transformer.transform(rng.normal(size=(3, DIM))))
            assert output.shape == (3, DIM)
        except NotImplementedError:
            pass


class Pipeline(DecoupledTransformer):
    def __init__(self, steps=(), *, input_streams=None, reroute_inputs=True, output_streams=None, log_level=None):
        self.steps: list[DecoupledTransformer] = steps
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
        self.partial_fit_transform(data, stream)

    def _partial_fit_transform(self, data, stream=0, return_output_stream=False):
        stream = self.input_streams[stream]
        for step in self.steps:
            data, stream = step.partial_fit_transform(data, stream=stream, return_output_stream=True)

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


class TypicalTransformer(DecoupledTransformer):
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


class CenteringTransformer(TypicalTransformer):
    def __init__(self, *, init_size=0, input_streams=None, output_streams=None, on_nan_width=None, log_level=None):
        super().__init__(input_streams=input_streams, output_streams=output_streams, on_nan_width=on_nan_width, log_level=log_level)
        self.init_size = init_size
        self.samples_seen = 0
        self.center = 0

    def pre_initialization_fit_for_X(self, X):
        self.partial_fit_for_X(X)
        if self.samples_seen >= self.init_size:
            self.is_initialized = True

    def partial_fit_for_X(self, X):
        self.samples_seen += X.shape[0]
        self.center = self.center + (X.sum(axis=0) - X.shape[0] * self.center) / self.samples_seen

    def transform_for_X(self, X):
        return X - self.center

    def inverse_transform_for_X(self, X):
        return X + self.center

    def instance_get_params(self, deep=True):
        return {'init_size': self.init_size}

class ZScoringTransformer(TypicalTransformer):
    # see https://math.stackexchange.com/a/1769248/701602
    """
    Examples
    --------
    >>> X = np.random.normal(size=(1000, 5)) * np.arange(5)
    >>> z = ZScoringTransformer(freeze_after_init=False)
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


class KernelSmoother(StreamingTransformer):
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

    def _partial_fit_transform(self, data, stream, return_output_stream):
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
        return data, stream if return_output_stream else data

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



class Concatenator(StreamingTransformer):

    def __init__(self, *, input_streams=None, output_streams=None, log_level=None, stream_scaling_factors=None):
        input_streams = input_streams or PassThroughDict({0:0, 1:1})

        output_stream = max(input_streams.keys()) + 1
        output_streams = output_streams or PassThroughDict({k: output_stream for k in input_streams.keys()} | {'skip': -1})
        super().__init__(input_streams=input_streams, output_streams=output_streams, log_level=log_level)
        self.last_seen = {}

        if stream_scaling_factors is None:
            stream_scaling_factors = {i:1 for i in self.input_streams}

        self.stream_scaling_factors = stream_scaling_factors

    def _partial_fit_transform(self, data, stream, return_output_stream):
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


class Tee(DecoupledTransformer):
    def __init__(self, input_streams=None, log_level=None, output_streams=None):
        input_streams = input_streams or PassThroughDict()
        self.observed = {}
        super().__init__(input_streams=input_streams, log_level=log_level, output_streams=output_streams)

    def _partial_fit(self, data, stream):
        if stream in self.input_streams:
            if stream not in self.observed:
                self.observed[stream] = []
            self.observed[stream].append(data)

    def transform(self, data, stream=0, return_output_stream=False):
        return (data, stream) if return_output_stream else data

    def convert_to_array(self):
        self.observed = {k: ArrayWithTime.from_list(v, squeeze_type='to_2d', drop_early_nans=True) for k, v in self.observed.items()}
        return self.observed