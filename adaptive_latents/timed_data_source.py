from abc import ABC, abstractmethod
import numpy as np
import types
import warnings


class DataSource(ABC):
    def __new__(cls, source=None, *args, **kwargs):
        # todo: is source=none a bad idea? it's to get deepcopy working
        if isinstance(source, GeneratorDataSource) or hasattr(source, 'next_sample_time'):
            # TODO: choose duck typing or inheritance checking here
            return source
        return super().__new__(cls)

    @abstractmethod
    def next_sample_time(self) -> float:
        pass

    @abstractmethod
    def current_sample_time(self) -> float:
        pass

    @property
    @abstractmethod
    def dt(self) -> float:
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
        a = np.array(source)
        if len(a.shape) == 1:
            a = a[:, None]
        if len(a.shape) == 2:
            a = a[:, None, :]

        # assert a.shape[0] * a.shape[1] > a.shape[2]

        self.a = a
        self.t = timepoints if timepoints is not None else np.arange(a.shape[0])
        assert len(self.t) == len(self.a)

        self.index = 0

    def __iter__(self):
        # TODO: maybe this class shouldn't be both an iterable and its own iterator
        self.index = 0
        return self

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

    @property
    def dt(self):
        dts = np.diff(self.t)
        dt = np.median(dts)
        assert np.ptp(dts)/dt < 0.05
        return dt

    @staticmethod
    def from_nwb_timeseries(timeseries):
        return NumpyTimedDataSource(timeseries.data[:], timeseries.timestamps[:])


class ArrayWithTime(np.ndarray):
    "The idea is to subclass here, but it seems pretty involved."
    # https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    # https://stackoverflow.com/a/51955094
    def __new__(cls, input_array, t):
        obj = np.asarray(input_array).view(cls)
        obj.t = t
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        # assert len(self.shape) == 3

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
    def from_NTDS(cls, ds):
        return cls(np.squeeze(ds.a, axis=2), ds)

    @classmethod
    def from_transformed_data(cls, new_data, old_data):
        # refers to the outputs of a transformer
        new_data = np.array(new_data)
        if hasattr(old_data, 't'):
            return cls(new_data, old_data.t)
        else:
            return new_data
