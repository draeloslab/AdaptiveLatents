import copy
from abc import ABC, abstractmethod
from .timed_data_source import DataSource, GeneratorDataSource, NumpyTimedDataSource
from frozendict import frozendict
import numpy as np
import types
from collections import deque


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
    def __init__(self, input_streams=None, output_streams=None, log_level=0, **kwargs):
        """
        Parameters
        ----------
        input_streams: dict[int, str]
            Keys are stream numbers, values are which semantic block they correspond to.
            So {3: 'X'} would mean that stream 3 should be processed as an X variable.
            Data not in an input_stream will usually be passed through.
        output_streams: dict[int, int]
            Keys are input streams, values are output streams; this is stream remapping applied after the transformer.
        """

        # TODO: get consistent printing
        self.kwargs = kwargs  # mostly for printing later, these sometimes correspond to the estimator; this works when being used as a mixin
        super().__init__(**kwargs)
        # if input_streams is not None:
        #     self.kwargs.update(input_streams=input_streams)
        # if output_streams is not None:
        #     self.kwargs.update(output_streams=output_streams)

        self.input_streams = PassThroughDict(input_streams or {0: 'X'})
        self.output_streams = PassThroughDict(output_streams or {})
        self.log_level = log_level
        self.log = dict()

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

        ret = self._partial_fit_transform(data, stream, return_output_stream)
        self.log_for_partial_fit(data, stream)
        return ret

    def log_for_partial_fit(self, data, stream):
        pass

    @abstractmethod
    def _partial_fit_transform(self, data, stream, return_output_stream):
        pass

    def get_params(self, deep=True):
        if deep:
            return dict(input_streams=self.input_streams, output_streams=self.output_streams, log_level=self.log_level)
        else:
            return dict()

    def trace_route(self, stream):
        middle_str = str(self) if stream in self.input_streams else ""
        if stream == self.output_streams[stream]:
            return middle_str
        return [stream, middle_str, self.output_streams[stream]]


    def streaming_run_on(self, sources, return_output_stream=False):
        """
        Parameters
        ----------
        sources: np.ndarray, types.GeneratorType, list[np.ndarray | types.GeneratorType], DataSource, list[DataSource], list[tuple[DataSource, int]]
            This should be the set of data sources.
            If a single DataSource, it will be promoted to [streams]
            If a list of DataSources, each source will be mapped to the equal to its index in the list.
            If a list of tuples, the first element of each tuple will be mapped to the stream number in the second element.
        return_output_stream: bool
            Whether to yield the output stream or not. This is false by default to not confuse first-time users.

        Yields
        -------
        data: np.ndarray
            The processed version of each element of the given iterator.
        stream: int, optional
            the stream that the outputted data belongs to
        """

        if not (isinstance(sources, tuple) or isinstance(sources, list)):  # passed a single source
            sources = [sources]
        elif not len(sources): # passed an empty list
            return

        if not isinstance(sources[0], tuple):  # passed a list of sources without streams
            streams = range(len(sources))
            sources = zip(sources, streams)

        sources, streams = zip(*sources)

        sources = [NumpyTimedDataSource(source) if isinstance(source, np.ndarray) else GeneratorDataSource(source) for source in sources]

        sources = list(zip(map(iter, sources), streams))

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


    def offline_run_on(self, sources, convinient_return=True):
        outputs = {}
        for data, stream in self.streaming_run_on(sources, return_output_stream=True):
            outputs[stream] = outputs.get(stream, []) + [data]

        if convinient_return:
            if 0 not in outputs:
                raise Exception("No outputs were routed to stream 0.")

            data = outputs[0]
            while data and np.isnan(data[0]).any():
                data.pop(0)
            outputs = np.squeeze(data)

        return outputs

    def __str__(self):
        kwargs = ', '.join(f'{k}={v}' for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({kwargs})"


class DecoupledTransformer(StreamingTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen = False

    def partial_fit_transform(self, data, stream=0, return_output_stream=False):
        self.partial_fit(data, stream)
        return self.transform(data, stream, return_output_stream)

    def partial_fit(self, data, stream=0):
        if self.frozen:
            return
        self._partial_fit(data, stream)
        self.log_for_partial_fit(data, stream)

    def _partial_fit_transform(self, data, stream, return_output_stream):
        raise NotImplementedError()

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

    def inverse_transform(self, data, stream=0, return_output_stream=False):
        raise NotImplementedError()



class Pipeline(DecoupledTransformer):
    def __init__(self, steps=(), input_streams=None, **kwargs):
        self.steps: list[DecoupledTransformer] = steps

        expected_streams = set(k for step in self.steps for k in step.input_streams.keys())
        self.expected_streams = sorted(expected_streams, key=lambda x: str(x))
        if input_streams is None:
            input_streams = dict(zip(range(len(self.expected_streams)), self.expected_streams))

        super().__init__(input_streams, **kwargs)

    def get_params(self, deep=True):
        # TODO: make this actually SKLearn compatible
        p = dict(steps=[x for x in self.steps])
        if deep:
            for i, step in enumerate(self.steps):
                for k, v in step.get_params(deep).items():
                    p[f'__steps[{i}]__{k}'] = v
        return p | super().get_params(deep)

    def _partial_fit(self, data, stream=0):
        self.partial_fit_transform(data, stream)

    def partial_fit_transform(self, data, stream=0, return_output_stream=False):
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
        kwargs = ' '.join(f'{k}={v}' for k, v in self.kwargs.items())
        return f"{self.__class__.__name__}([{', '.join(str(s) for s in self.steps)}]{', ' + kwargs if kwargs else ''})"


class TypicalTransformer(DecoupledTransformer):
    def __init__(self, input_streams=None, output_streams=None, **kwargs):
        input_streams = input_streams or {0: 'X'}
        super().__init__(input_streams, output_streams, **kwargs)
        self.is_initialized = False

    def get_params(self, deep=True):
        p = super().get_params(deep)
        p = self.instance_get_params() | p
        return p

    def _partial_fit(self, data, stream=0):
        if self.input_streams[stream] == 'X':
            if np.isnan(data).any():
                idx = np.isnan(data).any(axis=1)
                if idx.all():
                    return
                data = data[np.isnan(data).any(axis=1)]

            if not self.is_initialized:
                self.pre_initialization_fit_for_X(data)
            else:
                self.partial_fit_for_X(data)

    def transform(self, data, stream=0, return_output_stream=False):
        if self.input_streams[stream] == 'X':
            if not self.is_initialized or np.isnan(data).any():
                data = np.nan * data
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

    def freeze(self, b=True):
        self.frozen = b

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_initialized = True
        self.samples_seen = 0
        self.center = 0

    def partial_fit_for_X(self, X):
        if not self.frozen:
            self.samples_seen += X.shape[0]
            self.center = self.center + (X.sum(axis=0) - X.shape[0] * self.center) / self.samples_seen

    def transform_for_X(self, X):
        return X - self.center

    def inverse_transform_for_X(self, X):
        return X + self.center

    def instance_get_params(self, deep=True):
        return {}


class KernelSmoother(TypicalTransformer):
    # TODO: make time aware
    # TODO: make a StreamingTransformer
    def __init__(self, tau=1, kernel_length=None, custom_kernel=None, **kwargs):
        super().__init__(**kwargs)
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

    def instance_get_params(self, deep=True):
        return dict(tau=self.tau, kernel_length=self.kernel_length, custom_kernel=self.custom_kernel)

    def pre_initialization_fit_for_X(self, X):
        if self.last_X is not None:
            self.history.extend(self.last_X)
        self.last_X = X.copy()
        if len(self.history) == len(self.kernel):
            self.is_initialized = True

    def partial_fit_for_X(self, X):
        if not self.frozen:
            self.history.extend(self.last_X)
            self.last_X = X.copy()

    def transform_for_X(self, X):
        d = self.history.copy()
        new_X = np.empty_like(X)  # otherwise we modify the array in-place
        for i, row in enumerate(X):
            d.append(row)
            new_X[i] = self.kernel @ d
        return new_X

    def plot_impulse_response(self, ax):
        """
        Parameters
        ----------
        ax: matplotlib.pyplot.Axes
            The axis to plot on.
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


# class Concatenator(TransformerMixin):
#     def __init__(self, input_streams, output_streams):
#         super().__init__(input_streams, output_streams)
#         self.last_seen = {k: None for k in self.input_streams.keys()}
#
#     def partial_fit_transform(self, data, stream=0):
#         self.partial_fit(data, stream)
#         return self.transform(data, stream)
#
#     def partial_fit(self, data, stream=0):
#         if stream in self.input_streams:
#             self.last_seen[self.input_streams[stream]] = data
#
#     def transform(self, data, stream=0):
#         if stream in self.input_streams:
#             last_seen = dict(self.last_seen)
#             last_seen[self.input_streams[stream]] = data
#             values = filter(lambda x: x is not None, last_seen.values())
#             return np.hstack(list(values))
#         else:
#             return data
