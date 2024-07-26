from abc import ABC, abstractmethod
from adaptive_latents.input_sources.timed_data_source import DataSource, GeneratorDataSource, NumpyTimedDataSource
from frozendict import frozendict
import numpy as np
import types


class PassThroughDict(frozendict):
    def __missing__(self, key):
        return key


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
        self.kwargs = kwargs  # mostly for printing later
        super().__init__(**kwargs)
        # if input_streams is not None:
        #     self.kwargs.update(input_streams=input_streams)
        # if output_streams is not None:
        #     self.kwargs.update(output_streams=output_streams)

        self.input_streams = PassThroughDict(input_streams or {0: 'X'})
        self.output_streams = PassThroughDict(output_streams or {})
        self.pipeline_post_fit_hooks = []
        self.log_level = log_level

    @abstractmethod
    def partial_fit(self, data, stream=0):
        """data should be of shape (n_samples, sample_size)"""
        pass

    @abstractmethod
    def transform(self, data, stream=0, return_output_stream=False):
        """data should be of shape (n_samples, sample_size)"""
        pass

    def partial_fit_transform(self, data, stream=0, return_output_stream=False):
        self.partial_fit(data, stream)
        return self.transform(data, stream, return_output_stream)

    def log_for_step(self):
        pass

    def generator_fit_transform(self, sources, return_output_stream=False):
        """
        Parameters
        ----------
        sources: np.ndarray, types.GeneratorType, list[np.ndarray | types.GeneratorType], DataSource, list[DataSource], list[tuple[DataSource, int]]
            This should be the set of data sources.
            If a single DataSource, it will be promoted to [streams]
            If a list of DataSources, each source will be mapped to the equal to its index in the list.
            If a list of tuples, the first element of each tuple will be mapped to the stream number in the second element.
        return_output_stream: bool
            Wheither to return the output stream or not. This is false by default to not confuse first-time users.

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

        sources = [GeneratorDataSource(source) for source in sources]

        sources = list(zip(sources, streams))

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

        # for X in iterable:
        #     X = self.partial_fit_transform(X)
        #     yield X

    def offline_fit_transform(self, sources, convinient_return=True):
        outputs = {}
        for data, stream in self.generator_fit_transform(sources, return_output_stream=True):
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
    def __init__(self, steps, input_streams=None):
        self.steps: list[TransformerMixin] = steps

        expected_streams = set(k for step in self.steps for k in step.input_streams.keys())
        self.expected_streams = sorted(expected_streams, key=lambda x: str(x))
        if input_streams is None:
            input_streams = dict(zip(range(len(self.expected_streams)), self.expected_streams))

        super().__init__(input_streams)

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

        if not return_output_stream:
            return data
        return data, self.output_streams[stream]

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

        if return_output_stream:
            return data, self.output_streams[stream]
        return data

    def pre_initialization_fit_for_X(self, X):
        self.is_initialized = True

    @abstractmethod
    def partial_fit_for_X(self, X):
        pass

    @abstractmethod
    def transform_for_X(self, X):
        pass


class CenteringTransformer(TypicalTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_initialized = True
        self.samples_seen = 0
        self.center = 0

    def partial_fit_for_X(self, X):
        self.samples_seen += X.shape[0]
        self.center = self.center + (X.sum(axis=0) - X.shape[0] * self.center) / self.samples_seen

    def transform_for_X(self, X):
        return X - self.center


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
