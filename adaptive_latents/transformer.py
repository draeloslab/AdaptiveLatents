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



class TransformerMixin(ABC):
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
        self.kwargs = kwargs  # mostly for printing later, these sometimes correspond to the estimator; this works when being used as a mixin
        super().__init__(**kwargs)
        # if input_streams is not None:
        #     self.kwargs.update(input_streams=input_streams)
        # if output_streams is not None:
        #     self.kwargs.update(output_streams=output_streams)

        self.input_streams = PassThroughDict(input_streams or {0: 'X'})  # TODO: remove this default
        self.output_streams = PassThroughDict(output_streams or {})
        self.log_level = log_level
        self.log = dict()
        self.mid_run_sources = None  # todo: remove this?
        self.frozen = False

    @abstractmethod
    def partial_fit(self, data, stream=0):
        """data should be of shape (n_samples, sample_size)"""
        pass

    @abstractmethod
    def transform(self, data, stream=0, return_output_stream=False):
        """
        Applies a learned transformation to incoming data.

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
        pass

    def inverse_transform(self, data, stream=0, return_output_stream=False):
        raise NotImplementedError()

    # TODO: this is abstract to force developers to remember to use "freeze", but that's is also covered in the
    #  tests; maybe delete it?
    @abstractmethod
    def freeze(self, b=True):
        self.frozen = b

    def partial_fit_transform(self, data, stream=0, return_output_stream=False):
        self.partial_fit(data, stream)
        return self.transform(data, stream, return_output_stream)

    def log_for_partial_fit(self, data, stream=0):
        pass

    def run_on(self, sources, fit=True, return_output_stream=False):
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
        fit: bool
            Determines if fit_transform or fit is called.

        Yields
        -------
        data: np.ndarray
            The processed version of each element of the given iterator.
        stream: int, optional
            the stream that the outputted data belongs to
        """
        if not (isinstance(sources, tuple) or isinstance(sources, list)): # passed a single source
            sources = [sources]

        if not isinstance(sources[0], tuple):  # passed a list of sources without streams
            streams = range(len(sources))
            sources = zip(sources, streams)

        sources, streams = zip(*sources)

        # TODO: think more about if this should be a deep copy or not
        # or if NumpyTimedDataSource should not be both the data structure and iterator
        sources = [NumpyTimedDataSource(copy.deepcopy(source)) if isinstance(source, np.ndarray) else GeneratorDataSource(source) for source in sources]

        sources = list(zip(sources, streams))
        self.mid_run_sources = sources

        while True:  # while-true/break is a code smell, but I want a do-while
            next_time = float('inf')
            for source, stream in reversed(sources):  # reversed to prefer the first element
                source_next_time = source.next_sample_time()
                if source_next_time <= next_time:
                    next_time = source_next_time
                    next_source, next_stream = source, stream
            if not next_time < float('inf'):
                self.mid_run_sources = None # todo: GeneratorExit exception cleanup?
                break
            if fit:
                ret = self.partial_fit_transform(data=next(next_source), stream=next_stream, return_output_stream=return_output_stream)
            else:
                ret = self.transform(data=next(next_source), stream=next_stream, return_output_stream=return_output_stream)

            yield ret

        self.mid_run_sources = []


    def offline_run_on(self, sources, fit=True, convinient_return=True):
        outputs = {}
        for data, stream in self.run_on(sources, fit=fit, return_output_stream=True):
            outputs[stream] = outputs.get(stream, []) + [data]

        if convinient_return:
            data = outputs[0]
            while data and np.isnan(data[0]).any():
                data.pop(0)
            return np.squeeze(data)
        else:
            return outputs

    def __str__(self):
        kwargs = ', '.join(f'{k}={v}' for k, v in self.kwargs.items())
        return f"{self.__class__.__name__}({kwargs})"

    def trace_route(self, stream):
        middle_str = str(self) if stream in self.input_streams else ""
        if stream == self.output_streams[stream]:
            return middle_str
        return [stream, middle_str, self.output_streams[stream]]


class Pipeline(TransformerMixin):
    def __init__(self, steps, input_streams=None, **kwargs):
        self.steps: list[TransformerMixin] = steps

        expected_streams = set(k for step in self.steps for k in step.input_streams.keys())
        self.expected_streams = sorted(expected_streams, key=lambda x: str(x))
        if input_streams is None:
            input_streams = dict(zip(range(len(self.expected_streams)), self.expected_streams))

        super().__init__(input_streams, **kwargs)

    def partial_fit(self, data, stream=0):
        self.partial_fit_transform(data, stream)

    def partial_fit_transform(self, data, stream=0, return_output_stream=False):
        stream = self.input_streams[stream]
        for step in self.steps:
            data, stream = step.partial_fit_transform(data, stream=stream, return_output_stream=True)

        if not return_output_stream:
            return data
        return data, self.output_streams[stream]

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


class TypicalTransformer(TransformerMixin):
    def __init__(self, input_streams=None, output_streams=None, **kwargs):
        input_streams = input_streams or {0: 'X'}
        super().__init__(input_streams, output_streams, **kwargs)
        self.is_initialized = False

    def partial_fit(self, data, stream=0):
        if self.frozen:
            return
        if self.input_streams[stream] == 'X':
            if np.isnan(data).any():
                idx = np.isnan(data).any(axis=1)
                if idx.all():
                    return
                data = data[np.isnan(data).any(axis=1)]

            if not self.is_initialized:
                self.pre_initialization_fit_for_X(data)
                self.log_for_partial_fit(data, pre_initialization=True)
            else:
                self.partial_fit_for_X(data)
                self.log_for_partial_fit(data)

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

    def inverse_transform_for_X(self, X):
        raise NotImplementedError()

    def log_for_partial_fit(self, data, stream=0, pre_initialization=False):
        pass



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

    def freeze(self, b=True):
        self.frozen = b


class KernelSmoother(TypicalTransformer):
    # TODO: make time aware
    def __init__(self, tau=1, kernel_length=None, custom_kernel=None, **kwargs):
        super().__init__(**kwargs)
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

    def freeze(self, b=True):
        self.frozen = b

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
