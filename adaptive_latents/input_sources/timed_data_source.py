import numpy as np


class NumpyTimedDataSource:
    def __init__(self, a, timepoints=None):
        if isinstance(a, NumpyTimedDataSource):
            timepoints = a.t
            a = a.a
        a = np.array(a)
        if len(a.shape) == 1:
            a = a[:, None]
        assert a.shape[0] > a.shape[1]

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
        if self.index >= len(self.t):
            return float('inf')
        return self.t[self.index + 1]