from abc import ABC, abstractmethod
import numpy as np
from collections import deque
from proSVD import proSVD
from .functional import get_from_saved_npz

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hmm_simulation import HMM



class DataSource(ABC):
    def __init__(self, output_shape, time_offsets=()):
        self.length = None
        self.time_offsets = time_offsets
        self.output_shape = output_shape
        self.init_size = 0

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def get_atemporal_data_point(self, offset=0):
        pass

    @abstractmethod
    def get_history(self, depth=None):
        pass

    def __len__(self):
        return self.length

    # @abstractmethod
    # def reset(self):
    #     pass


class ConsumableDataSource(DataSource, ABC):
    def triples(self, limit=None):
        count = 0
        for obs, beh in self:
            pairs = {}
            for offset in self.time_offsets:
                pairs[offset] = self.get_atemporal_data_point(offset)
            if count == limit:
                return
            yield obs, beh, pairs
            count += 1


# class PairWrapperSource(ConsumableDataSource):
#     def shorten(self, n):
#         self.obs.shorten(n)
#         self.beh.shorten(n)
#
#     def __init__(self, obs, beh):
#         self.obs: DataSource = obs
#         self.beh: DataSource = beh
#
#         bigger_init = max(self.obs.init_size, self.beh.init_size)
#         self.obs.shorten(bigger_init - self.obs.init_size)
#         self.beh.shorten(bigger_init - self.beh.init_size)
#
#         assert self.obs.time_offsets == self.beh.time_offsets
#         super().__init__(output_shape=(obs.output_shape, beh.output_shape), time_offsets=self.obs.time_offsets)
#
#         assert len(obs) == len(beh)
#         self.length = len(obs)
#
#         self.index = 0
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         return next(self.obs), next(self.beh)
#
#     def get_atemporal_data_point(self, offset=0):
#         return self.obs.get_atemporal_data_point(offset), self.beh.get_atemporal_data_point(offset)
#
#     def get_history(self, depth=None):
#         return self.obs.get_history(), self.beh.get_history()


# class ConcatenatorSource(DataSource):
#     def __init__(self, inputs):
#         self.inputs: list[DataSource] = inputs
#         output_shape = 0
#         for input in self.inputs:
#             assert isinstance(input.output_shape, int)
#             assert input.time_offsets == inputs[0].time_offsets
#             assert len(input) == len(inputs[0])
#             output_shape += input.output_shape
#         super().__init__(output_shape, time_offsets=inputs[0].time_offsets)
#         self.length = len(inputs[0])
#
#     def __next__(self):
#         nexts = [next(i) for i in self.inputs]
#         return np.hstack(nexts)
#
#     def get_atemporal_data_point(self, offset=0):
#         pts = [i.get_atemporal_data_point(offset) for i in self.inputs]
#         return np.hstack(pts)
#
#     def get_history(self, depth=None):
#         pts = [i.get_history(depth) for i in self.inputs]
#         return np.hstack(pts)
#
#     def shorten(self, n):
#         for i in self.inputs:
#             i.shorten(n)
#         self.length -= n
#
#     def __iter__(self):
#         return self


# class SingleStreamDataSource(DataSource, ABC):
#     def __init__(self, output_shape, time_offsets=(), min_memory_radius=100_000):
#         super().__init__(output_shape, time_offsets)
#
#         self.necessary_buffer = [-1 * min(min(time_offsets, default=0), 0) + 1,
#                                  max(max(time_offsets, default=0), 0)]  # +1 for current state
#         self.memory_radius = max(min_memory_radius, 2 * max([abs(t) for t in time_offsets], default=0))
#
#         self.future = deque(maxlen=self.memory_radius + self.necessary_buffer[1])
#         self.past = deque(maxlen=self.memory_radius + self.necessary_buffer[0])
#         # present is past[0], 1 step in the future is future[0]
#         self.index = 0
#
#     @abstractmethod
#     def simulate_more_steps(self):
#         pass
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.index >= self.length:
#             raise StopIteration
#         if len(self.future) < self.necessary_buffer[1]:
#             self.simulate_more_steps()
#
#         if len(self.future) < self.necessary_buffer[1]:
#             raise Exception("self.length should protect from this")
#
#         current = self.future.popleft()
#         self.past.appendleft(current)
#
#         self.index += 1
#         return current
#
#     def get_atemporal_data_point(self, offset=0):
#         if offset <= 0:
#             return self.past[-offset]
#         if offset > 0:
#             if len(self.future) <= offset:
#                 self.simulate_more_steps()
#             return self.future[offset - 1]
#
#     def get_history(self, depth=None):
#         # todo: this is a hack
#         p = self.past
#         if p and isinstance(p[0], tuple):
#             p = [np.array(a) for a in zip(*p)]
#         return p
#
#     def shorten(self, n):
#         for _ in range(n):
#             next(self)
#         self.length -= n


# class HMMSimDataSourceSingle(SingleStreamDataSource, ConsumableDataSource):
#     def __init__(self, hmm, seed, length, time_offsets=()):
#         super().__init__(output_shape=(hmm.emission_model.embedded_dimension, 1), time_offsets=time_offsets)
#
#         self.hmm: HMM = hmm
#         self.seed = seed
#         self.rng = np.random.default_rng(seed)
#
#         self.length = length
#
#         beh, obs = self.hmm.simulate_with_states(self.necessary_buffer[0], self.rng)
#         for i in range(obs.shape[0]):
#             self.past.appendleft((obs[i], beh[i]))
#         self.current_state = beh[-1]
#
#         self.simulate_more_steps()
#
#     def simulate_more_steps(self):
#         n = self.memory_radius - len(self.future)
#         beh, obs = self.hmm.simulate_with_states(n, self.rng, self.current_state)
#         self.current_state = beh[-1]
#
#         for i in range(obs.shape[0]):
#             self.future.append((obs[i], beh[i]))


class NumpyTimedDataSource(DataSource):
    def __init__(self, a, timepoints, time_offsets=()):
        a = np.array(a)
        if len(a.shape) == 1:
            a = a[:, None]
        super().__init__(output_shape=len(a[0]), time_offsets=time_offsets)
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

    def current_timepoint(self):
        return self.t[self.index]

    def preview_next_timepoint(self):
        return self.t[self.index + 1], self.index + 1 >= len(self)

    #
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

        return self.a[slice_start:slice_end]


class NumpyPairedDataSource(ConsumableDataSource):
    def __init__(self, obs, beh=np.array([]), time_offsets=()):
        self.obs = obs
        self.beh = beh
        if len(beh.shape) == 1:
            self.beh = beh.reshape((obs.shape[0], -1))

        super().__init__(output_shape=(self.obs.shape[1], self.beh.shape[1]), time_offsets=time_offsets)

        assert len(self.beh) == len(self.obs)

        self.clear_range = (0, len(obs))
        if time_offsets:
            self.clear_range = (max(0, -min(time_offsets)), len(obs) - max(max(time_offsets), 0))
        self.length = self.clear_range[1] - self.clear_range[0]

        self.index = 0

    def shorten(self, n):
        self.clear_range = (self.clear_range[0] + n, self.clear_range[1])
        self.length = self.clear_range[1] - self.clear_range[0]

    def __iter__(self):
        return self

    def __next__(self):
        try:
            o, b = self._get_pair(self.index, offset=0)
        except IndexError:
            raise StopIteration()

        self.index += 1
        return o, b

    def get_atemporal_data_point(self, offset=0):
        """gets a data pair relative to the present pair"""
        return self._get_pair(item=self.index - 1, offset=offset)

    def _get_pair(self, item, offset=0):
        # could be __getitem__

        if item < 0:
            raise IndexError("Negative indexes are not supported.")

        if item >= len(self):
            raise IndexError("Index out of range.")

        inside_index = item + self.clear_range[0] + offset
        return self.obs[inside_index, :], self.beh[inside_index, :]

    def get_history(self, depth=None):
        slice_end = self.index + self.clear_range[0]
        slice_start = self.clear_range[0]
        if depth is not None:
            slice_start = slice_end - depth

        if slice_start < self.clear_range[0]:
            raise IndexError()

        o = self.obs[slice_start:slice_end, :]
        b = self.beh[slice_start:slice_end, :]
        return o, b


# class ProSVDDataSourceSingle(SingleStreamDataSource):
#     def __init__(self, input_source, output_d, init_size=100, min_memory_radius=100_000, time_offsets=()):
#         super().__init__(output_shape=output_d, time_offsets=time_offsets, min_memory_radius=min_memory_radius)
#         self.output_d = output_d
#         self.input_source: DataSource = input_source
#         assert len(self.input_source.time_offsets) == 0
#         self.pro = proSVD(k=output_d)
#
#         l = []
#         for _ in range(init_size):
#             obs = next(self.input_source)
#             l.append(obs)
#         self.pro.initialize(np.array(l).T)
#
#         for i in range(self.necessary_buffer[0]):
#             obs = next(self.input_source)
#             self.pro.preupdate()
#             self.pro.updateSVD(obs[:, None])
#             self.pro.postupdate()
#
#             obs = obs @ self.pro.Q
#
#             self.past.appendleft(obs)
#
#         self.length = len(input_source) - init_size - sum(self.necessary_buffer)
#         self.init_size = init_size + 1
#
#         assert self.length >= 0
#
#         self.simulate_more_steps()
#
#     def simulate_more_steps(self):
#         # todo: might be better to do one at a time
#         n = self.memory_radius - len(self.future)
#
#         for _ in range(n):
#             try:
#                 obs = next(self.input_source)
#             except StopIteration:
#                 break
#             self.pro.preupdate()
#             self.pro.updateSVD(obs[:, None])
#             self.pro.postupdate()
#
#             obs = obs @ self.pro.Q
#             self.future.append(obs)
