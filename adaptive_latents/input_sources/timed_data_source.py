from abc import ABC, abstractmethod
import numpy as np
import types
import warnings


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


class _NumpyTimedDataSource(np.ndarray):
    "The idea is to subclass here, but it seems pretty involved."
    # https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    # https://stackoverflow.com/a/51955094
    def __new__(cls, input_array, t=None):
        obj = np.asarray(input_array).view(cls)
        obj.t = t
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        assert len(self.shape) == 3

        if hasattr(obj, 't'):
            self.t = obj.t
        else:
            self.t = np.arange(self.shape[0])

        if hasattr(obj, '_new_t_index'):
            self.t = self.t[obj._new_t_index]

    def __getitem__(self, item):

        if isinstance(item, (slice, int)):
            self._new_t_index = item
        else:
            self._new_t_index = item[0]
        
        return super().__getitem__(item)