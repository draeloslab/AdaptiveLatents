import types
from abc import ABC, abstractmethod

import numpy as np


class DataSource(ABC):
    @abstractmethod
    def next_sample_time(self) -> float:
        pass

    @abstractmethod
    def current_sample_time(self) -> float:
        pass


class GeneratorDataSource(DataSource):
    def __init__(self, source, dt=1):
        if isinstance(source, types.GeneratorType):
            generator = source
        else:
            generator = iter(source)
        self.generator = enumerate(generator)
        self.next_sample = next(self.generator)
        self._current_time = None
        self._dt = dt

    @property
    def dt(self):
        return self._dt

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

        self._current_time = this_sample[0]
        return ArrayWithTime(this_sample[1], t=self._current_time * self.dt)

    def next_sample_time(self):
        return self.next_sample[0]

    def current_sample_time(self):
        return self._current_time


class NumpyTimedDataSource(DataSource):
    def __init__(self, source, timepoints=None):
        self.a = source
        self.t = timepoints if timepoints is not None else np.arange(self.a.shape[0])
        assert len(self.t) == len(self.a)

        self.index = 0

    def __next__(self):
        try:
            d = self.a[self.index]
        except IndexError:
            raise StopIteration()

        d = ArrayWithTime(d.copy(), t=self.t[self.index])
        self.index += 1
        return d

    def next_sample_time(self):
        if self.index >= len(self.t):
            return float('inf')
        return self.t[self.index]

    def current_sample_time(self):
        if self.index == 0:
            return None
        return self.t[self.index-1]


class ArrayWithTime(np.ndarray):
    "The idea is to subclass here, but it seems pretty involved."
    # https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    # https://stackoverflow.com/a/51955094
    def __new__(cls, input_array, t):
        obj = np.asarray(input_array).view(cls)
        obj.t = np.array(t)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

        if hasattr(obj, 't'):
            self.t = obj.t

        # if hasattr(obj, '_new_t_index'):
        #     self.t = self.t[obj._new_t_index]

    # def __getitem__(self, item):
    #
    #     if isinstance(item, (slice, int)):
    #         self._new_t_index = item
    #     else:
    #         self._new_t_index = item[0]
    #
    #     return super().__getitem__(item)

    def __iter__(self):
        if hasattr(self.t, "__len__") and len(self.t) > 1 and len(self.t) == self.shape[0]:
            return NumpyTimedDataSource(self, self.t)
        else:
            return super().__iter__()

    def slice(self, start, stop=None):
        if isinstance(start, np.ndarray) or isinstance(start, slice):
            s = start
        else:
            s = slice(start, stop)
        return ArrayWithTime(self[s], self.t[s])

    def slice_by_time(self, s, *args):
        if not isinstance(s, slice):
            s = slice(s, *args)
        s: slice
        assert s.step in (1, None)

        if s.start is None:
            lower_constraint = True
        else:
            lower_constraint = (s.start <= self.t)

        if s.stop is None:
            upper_constraint = True
        else:
            upper_constraint = (self.t < s.stop)

        s_in_samples = lower_constraint & upper_constraint
        return ArrayWithTime(self[s_in_samples], self.t[s_in_samples])

    def as_array(self):
        return np.array(self)

    def time_to_sample(self, time):
        return np.searchsorted(self.t, time)

    @classmethod
    def align_indices(cls, a, b):
        # there's a faster way to do this with np.searchsorted
        a: cls
        assert (a.t[1:] - a.t[:-1] > 0).all()
        assert (b.t[1:] - b.t[:-1] > 0).all()
        idx_a = 0
        idx_b = 0
        a_indices = []
        b_indices = []

        while idx_a < len(a) and idx_b < len(b):
            d = a.t[idx_a] - b.t[idx_b]
            if np.isclose(0,d):
                a_indices.append(idx_a)
                b_indices.append(idx_b)
                idx_b += 1
                idx_a += 1
            elif d > 0:
                idx_b += 1
            else:
                idx_a += 1
        a_indices = np.array(a_indices)
        b_indices = np.array(b_indices)
        return cls(a[a_indices], a.t[a_indices]), cls(b[b_indices], b.t[b_indices])

    @classmethod
    def subtract_aligned_indices(cls, a, b):
        a, b = cls.align_indices(a, b)
        return cls(a - b, a.t)

    @property
    def dt(self):
        dts = np.diff(self.t)
        dt = np.median(dts)
        assert np.ptp(dts)/dt < 0.05
        return dt

    @classmethod
    def from_list(cls, input_list, squeeze_type='none', drop_early_nans=False):
        if drop_early_nans:
            i = 0
            while i < len(input_list) and not np.isfinite(input_list[i]).all():
                i += 1
            input_list = input_list[i:]

        t = np.array([x.t for x in input_list])
        if squeeze_type == 'none' or squeeze_type is None:
            input_array = np.array(input_list)
        elif squeeze_type == 'to_2d':
            input_array = np.squeeze(input_list)
            if len(input_array.shape) == 1:
                input_array = input_array[:, None]
            elif len(input_array.shape) == 3:
                # warnings.warn("squeezing 3d array to 2d, this is unusual")
                input_array = input_array.reshape([-1, input_array.shape[-1]])
            assert len(input_array.shape) == 2
        elif squeeze_type == 'squeeze':
            input_array = np.squeeze(input_list)
        else:
            raise ValueError()

        return cls(input_array=input_array, t=t)

    @classmethod
    def from_NTDS(cls, ds: NumpyTimedDataSource):
        return cls(np.squeeze(ds.a, axis=1), ds.t)

    @classmethod
    def from_transformed_data(cls, new_data, old_data):
        # refers to the outputs of a transformer
        new_data = np.array(new_data)
        if hasattr(old_data, 't'):
            return cls(new_data, old_data.t)
        else:
            return new_data

    @classmethod
    def from_nwb_timeseries(cls, timeseries):
        return cls(timeseries.data[:], timeseries.timestamps[:])


    @classmethod
    def from_notime(cls, a):
        return cls(a, np.arange(len(a)))

