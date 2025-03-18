from adaptive_latents.transformer import DecoupledTransformer
from adaptive_latents.timed_data_source import ArrayWithTime
from vjf.online import VJF as BaseVJF
import numpy as np
import torch


class VJF(DecoupledTransformer):
    base_algorithm = BaseVJF

    def __init__(self, *, config=None, latent_d=6, take_U=False, input_streams=None, output_streams=None, log_level=None):
        if input_streams is None:
            input_streams = {1: 'U'} if take_U else {}
            input_streams = input_streams | {0: 'Y', 2:'dt'}
        DecoupledTransformer.__init__(self=self, input_streams=input_streams, output_streams=output_streams,
                                      log_level=log_level)
        self.take_U = take_U
        self.latent_d = latent_d
        config = config or {}
        self.config = self.default_config_dict({'xdim': self.latent_d}) | config
        self.last_seen = {}
        self.vjf: BaseVJF | None = None
        self.q = None

    @staticmethod
    def default_config_dict(update=None):
        default_config = dict(
            resume=False,
            # xdim=6,  # dimension of hidden state
            udim=1,  # dimension of control vector
            # Ydim=udim,  # possibly not necessary?
            # Udim=udim,  # possibly not necessary?
            rdim=50,  # number of RBFs
            hdim=100,  # number of MLP hidden units
            lr=1e-3,  # learning rate
            clip_gradients=5.0,
            debug=False,
            likelihood='gaussian',  #
            system='rbf',
            recognizer='mlp',
            C=(None, True),  # loading matrix: (initial, estimate)
            b=(None, True),  # bias: (initial, estimate)
            A=(None, False),  # transition matrix if LDS
            Q=(1.0, True),  # state noise
            R=(1.0, True),  # observation noise
            random_seed=0,

            # these depend on the input dimensions
            # ydim=ydim,  # dimension of observations
            # B=(np.zeros((xdim, udim)), False),  # interaction matrix
        )

        update = update if update is not None else {}
        return default_config | update  # the | makes a copy of the original dict

    def init_vjf(self, ydim, udim=1):
        assert self.take_U or udim == 1

        self.config.update({
            'ydim': ydim,
            'udim': udim,
            'B': (np.zeros((self.config['xdim'], udim)), False),
        })

        self.vjf = BaseVJF(self.config)

    def _partial_fit(self, data, stream):
        if stream in self.input_streams:
            self.last_seen[self.input_streams[stream]] = data

            if len(self.last_seen) == len(self.input_streams):
                y, u = self.get_y_and_u()

                if self.vjf is None:
                    self.init_vjf(ydim=y.shape[-1], udim=u.shape[-1])

                y = torch.from_numpy(y).float()
                u = torch.from_numpy(u).float()
                self.q, _ = self.vjf.feed((y, u), q0=self.q)

    def transform(self, data, stream=0, return_output_stream=False):
        if self.input_streams[stream] == 'Y' and self.q is not None:
            q0 = self.q[0].detach().numpy()
            data = ArrayWithTime.from_transformed_data(q0, data)
        elif self.input_streams[stream] == 'dt' and self.q is not None:
            pass

        return (data, stream) if return_output_stream else data

    def get_y_and_u(self, y=None, u=None):
        if y is None:
            y = self.last_seen['Y']

        if u is None:
            if self.take_U:
                u = self.last_seen['U']
            else:
                u = np.zeros((y.shape[0], 1))

        return y, u

    def fit(self, y, u=None):
        y, u = self.get_y_and_u(y, u)
        if self.vjf is None:
            self.init_vjf(ydim=y.shape[-1], udim=u.shape[-1])

        mu, logvar, losses = self.vjf.filter(y[None, ...], u[None, ...])
        mu = mu.detach().numpy()[:, 0, :]
        logvar = logvar.detach().numpy()[:, 0, :]
        losses = torch.tensor(losses).detach().numpy()

        return mu, logvar, losses

    def get_params(self, deep=True):
        return super().get_params(deep=deep) | dict(take_U=self.take_U, latent_d=self.latent_d, config=self.config)