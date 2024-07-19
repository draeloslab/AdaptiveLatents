from abc import ABC, abstractmethod
from frozendict import frozendict


class PassThroughDict(frozendict):
    def __missing__(self, key):
        return key


class Transformer(ABC):
    def __init__(self, input_streams, output_streams=None):
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
        self.input_streams = PassThroughDict(input_streams)
        self.output_streams = PassThroughDict(output_streams or {})
        self.pipeline_post_fit_hooks = []

    @abstractmethod
    def partial_fit_transform(self, data, stream=0):
        """data should be of shape (n_samples, sample_size)"""
        pass

    @abstractmethod
    def transform(self, data, stream=0):
        """data should be of shape (n_samples, sample_size)"""
        pass

    def offline_fit(self, stream):
        for X in stream:
            self.partial_fit_transform(X)

    def offline_fit_transform(self, generator):
        for X in generator:
            X = self.partial_fit_transform(X)
            yield X

    def run_pipeline_post_fit_hooks(self):
        for hook in self.pipeline_post_fit_hooks:
            hook(caller=self)

    def __str__(self):
        return f"{self.__class__.__name__}"

    def trace_route(self, stream):
        middle_str = str(self) if stream in self.input_streams else ""
        if stream == self.output_streams[stream]:
            return middle_str
        return [stream, middle_str, self.output_streams[stream]]


class Add(Transformer):
    def __init__(self, to_add, input_streams=None, output_streams=None):
        input_streams = input_streams or {0: 'X'}
        super().__init__(input_streams, output_streams)
        self.to_add = to_add

    def partial_fit_transform(self, data, stream=0):
        return self.transform(data, stream)

    def transform(self, data, stream=0):
        if self.input_streams[stream] == 'X':
            data = data + self.to_add
        return data


class Concatenator(Transformer):
    def __init__(self, input_streams, output_streams):
        super().__init__(input_streams, output_streams)
        self.last_seen = {k: None for k in self.input_streams.keys()}

    def partial_fit_transform(self, data, stream=0):
        self.partial_fit(data, stream)
        return self.transform(data, stream)

    def partial_fit(self, data, stream=0):
        if stream in self.input_streams:
            self.last_seen[self.input_streams[stream]] = data

    def transform(self, data, stream=0):
        if stream in self.input_streams:
            last_seen = dict(self.last_seen)
            last_seen[self.input_streams[stream]] = data
            values = filter(lambda x: x is not None, last_seen.values())
            return np.hstack(list(values))
        else:
            return data

class Pipeline(Transformer):
    def __init__(self, steps, input_streams=None):
        self.steps: list[Transformer] = steps

        expected_streams = set(k for step in self.steps for k in step.input_streams.keys())
        self.expected_streams = sorted(expected_streams, key=lambda x: str(x))
        if input_streams is None:
            input_streams = dict(zip(range(len(self.expected_streams)), self.expected_streams))

        super().__init__(input_streams)

    def partial_fit_transform(self, data, stream=0):
        stream = self.input_streams[stream]
        for step in self.steps:
            data = step.partial_fit_transform(data, stream=stream)
            stream = step.output_streams.get(stream, stream)
            step.run_pipeline_post_fit_hooks()
        return data

    def transform(self, data, stream=0):
        stream = self.input_streams[stream]
        for step in self.steps:
            data = step.transform(data, stream=stream)
            stream = step.output_streams[stream]
        return data

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



if __name__ == '__main__':
    import numpy as np

    # a = Add(2)
    # print(a.transform(s, stream=2))


    s = np.zeros(6).reshape((2,3))
    p1 = Pipeline([
        Add(1),
        Add(1),
        Concatenator(input_streams={0: 'a', 1: 'b'}, output_streams={0: 0, 1: 0}),
        Add(1, input_streams={1:'X'}),
    ])
    p2 = Pipeline([
        p1,
        Add(2, output_streams={0:5}),
        p1,
    ])
    print(p2.trace_route(0))


    # ss = (s, s)
    #
    # print(a.transform(ss))
    #
    # print(p1.transform(ss))
    # #
    # # p2 = Pipeline([
    # #     p1,
    # #     Add(2)
    # # ])
    #
    # # sss = itertools.repeat((s, s), 2)
    # #
    # # for r in p1.transform(s):
    # #     print(r)