import numpy as np
import scipy.linalg
from adaptive_latents.transformer import TransformerMixin
from adaptive_latents.utils import column_space_distance

class BaseProPLS:
    def __init__(self, k=10, decay_alpha=1):
        self.k = k
        self.decay_alpha = decay_alpha

        self.u = None
        self.s = None
        self.vh = None
        self.n_samples_observed = 0

    def initialize(self, x, y):
        d_x = x.shape[1]
        d_y = y.shape[1]
        assert min(d_x, d_y) >= self.k, "k size doesn't make sense"

        self.u = np.zeros((d_x, self.k))
        self.s = np.zeros((self.k, self.k))
        self.vh = np.zeros((self.k, d_y))
        self.n_samples_observed = 0
        self.update(x, y) # TODO: check semantics of initialization

    def add_new_input_channels(self, n_x=0, n_y=0):
        if n_x:
            self.u = np.vstack([self.u, np.zeros(shape=(n_x, self.u.shape[1]))])
        if n_y:
            self.vh = np.hstack([self.vh, np.zeros((self.vh.shape[0], n_y))])

    def update(self, x, y):
        # decompose x into parallel and orthogonal components
        x_along = x @ self.u
        x_orth = x - x_along @ self.u.T
        r_x_orth, q_x_orth = scipy.linalg.rq(x_orth, mode='economic')

        # decompose y into parallel and orthogonal components
        y_along = y @ self.vh.T
        y_orth = y - y_along @ self.vh
        r_y_orth, q_y_orth = scipy.linalg.rq(y_orth, mode='economic')

        # decay old s information
        s_new = self.s * self.decay_alpha

        # construct the new svd
        u_new = np.hstack([self.u, q_x_orth.T])
        s_new = np.block([
            [s_new + x_along.T@y_along, x_along.T@r_y_orth],
            [r_x_orth.T@y_along, r_x_orth.T@r_y_orth]
        ])
        vh_new = np.vstack([self.vh, q_y_orth])

        # diagonalize the new svd
        u_rotation, s_new, vh_rotation = np.linalg.svd(s_new, full_matrices=False)

        # drop the smallest-covariance dimensions from our new svd
        u_new = u_new @ u_rotation[:,:self.k]
        s_new = np.diag(s_new[:self.k])
        vh_new = vh_rotation[:self.k] @ vh_new

        # align the new svd to the previous u and vh matrices with orthogonal procrustes
        temp = np.linalg.svd(u_new[:self.k])  # `u_new[:self.k] == u_new.T @ self.u`
        u_stabilizing_rotation = temp[0] @ temp[2]
        self.u = u_new @ u_stabilizing_rotation

        temp = np.linalg.svd(vh_new[:, :self.k])  # `vh_new[:, :self.k] == vh_new @ self.vh.T`
        vh_stabilizing_rotation = temp[2].T @ temp[0].T
        self.vh = vh_stabilizing_rotation @ vh_new

        self.s = u_stabilizing_rotation.T @ s_new @ vh_stabilizing_rotation.T

        # update the number of samples observed
        self.n_samples_observed *= self.decay_alpha
        self.n_samples_observed += x.shape[0]

    def project(self, *, x=None, y=None):
        x_proj = None
        y_proj = None

        if x is not None:
            x_proj = x @ self.u
        if y is not None:
            y_proj = y @ self.vh.T

        return tuple(filter(lambda z: z is not None, [x_proj, y_proj]))

    def get_cross_covariance(self):
        return self.u @ self.s @ self.vh


class proPLS(TransformerMixin, BaseProPLS):
    def __init__(self, input_streams=None, **kwargs):
        input_streams = input_streams or {0: 'X', 1: 'Y'}
        super().__init__(input_streams=input_streams,**kwargs)
        self.log = {'u': [], 'vh': [], 't': []}
        self.last_seen = {}
        self.is_initialized = False

    def partial_fit(self, data, stream=0):
        if self.frozen:
            return
        stream_label = self.input_streams[stream]
        if stream_label in ('X', 'Y'):
            if np.isnan(data).any():
                # TODO: you could be smarter about keeping certain rows, but I want to be correct first
                return

            if not self.is_initialized:
                self.last_seen[stream_label] = data
                if len(self.last_seen) == 2:
                    self.initialize(self.last_seen['X'], self.last_seen['Y'])
                    self.last_seen = {}
                    self.is_initialized = True
                self.log_for_partial_fit(data, pre_initialization=True)
            else:
                self.last_seen[stream_label] = data
                if len(self.last_seen) == 2:
                    self.update(self.last_seen['X'], self.last_seen['Y'])
                    self.log_for_partial_fit(data, stream)
                    self.last_seen = {}

    def transform(self, data, stream=0, return_output_stream=False):
        stream_label = self.input_streams[stream]
        if stream_label in ('X', 'Y'):
            if not self.is_initialized or np.isnan(data).any():
                data = np.nan * data
            else:
                if stream_label == 'X':
                    data = self.project(x=data)
                elif stream_label == 'Y':
                    data = self.project(y=data)

            self.log_for_transform(data)

        if return_output_stream:
            return data, self.output_streams[stream]
        return data

    def freeze(self, b=True):
        self.frozen = b

    def log_for_partial_fit(self, data, stream=0, pre_initialization=False):
        if not pre_initialization:
            if self.log_level > 0:
                # stream doesn't matter because they update at the same time
                self.log['u'].append(self.u)
                self.log['vh'].append(self.vh)
                self.log['t'].append(data.t)

    def log_for_transform(self, data, stream=0):
        pass

    def get_distance_from_subspace_over_time(self, subspace, variable='X'):
        assert self.log_level >= 1
        evolving_subspace = {
            'X': self.log['u'],
            'Y': map(np.transpose, self.log['vh']),
        }.get(variable)

        m = len(evolving_subspace)
        distances = np.empty(m)
        for j, Q in enumerate(evolving_subspace):
            if np.any(np.isnan(Q)):
                distances[j] = np.nan
                continue
            distances[j] = column_space_distance(Q, subspace)
        return distances, np.array(self.log['t'])