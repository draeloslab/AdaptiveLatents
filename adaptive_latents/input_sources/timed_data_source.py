import numpy as np

class NumpyTimedDataSource:
    def __init__(self, a, timepoints, time_offsets=()):
        a = np.array(a)
        if len(a.shape) == 1:
            a = a[:, None]
        assert a.shape[0] > a.shape[1]

        self.length = None
        self.time_offsets = time_offsets
        self.output_shape = len(a[0])
        self.init_size = 0



        self.a = a
        self.t = timepoints if timepoints is not None else np.arange(a.shape[0])
        assert len(self.t) == len(self.a)

        self.clear_range = (0, len(a))
        if self.time_offsets:
            self.clear_range = (max(0, -min(time_offsets)), len(a) - max(max(time_offsets), 0))

        self.length = int(self.clear_range[1] - self.clear_range[0])
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            d = self._get_item(self.index, offset=0)
        except IndexError:
            raise StopIteration()

        self.index += 1
        return d

    def __len__(self):
        return self.length

    def current_timepoint(self):
        return self.t[self.index]

    def preview_next_timepoint(self, offset = 1):
        return self.t[self.index + offset], self.index + offset >= len(self)

    def get_atemporal_data_point(self, offset=0):
        """gets a data pair relative to the present pair"""
        return self._get_item(item=self.index - 1, offset=offset)

    #
    def _get_item(self, item, offset=0):
        if item < 0:
            raise IndexError("Negative indexes are not supported.")

        if item >= len(self):
            raise IndexError("Index out of range.")

        inside_index = item + self.clear_range[0] + offset
        return self.a[inside_index]

    def get_history(self, depth=None):
        slice_end = self.index + self.clear_range[0]
        slice_start = self.clear_range[0]
        if depth is not None:
            slice_start = slice_end - depth

        if slice_start < self.clear_range[0]:
            raise IndexError()

        return self.a[slice_start:slice_end], self.t[slice_start:slice_end]

"""
todo:
    should these have state?
"""