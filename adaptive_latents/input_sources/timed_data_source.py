from abc import ABC, abstractmethod
import numpy as np
import types


class DataSource(ABC):
    def __new__(cls, source, *args, **kwargs):
        if isinstance(source, DataSource):
            return source
        return super().__new__(cls)

    @abstractmethod
    def next_sample_time(self):
        pass


class GeneratorDataSource(DataSource):
    def __init__(self, source):
        if isinstance(source, types.GeneratorType):
            generator = source
        else:
            generator = iter(source)
        self.generator = enumerate(generator)
        self.next_sample = tuple(next(self.generator))

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_sample[0] == float('inf'):
            raise StopIteration()

        this_sample = self.next_sample
        try:
            self.next_sample = next(self.generator)
        except StopIteration:
            self.next_sample = (float('inf'), None)

        return this_sample[1]

    def next_sample_time(self):
        return self.next_sample[0]


class NumpyTimedDataSource:
    def __init__(self, a, timepoints=None):
        a = np.array(a)
        if len(a.shape) == 1:
            a = a[:, None]
        if len(a.shape) == 2:
            a = a[:, None, :]

        assert a.shape[0] * a.shape[1] > a.shape[2]

        self.a = a
        self.t = timepoints if timepoints is not None else np.arange(a.shape[0])
        assert len(self.t) == len(self.a)

        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            d = self.a[self.index]
        except IndexError:
            raise StopIteration()

        self.index += 1
        return d

    def next_sample_time(self):
        if self.index+1 >= len(self.t):
            return float('inf')
        return self.t[self.index + 1]