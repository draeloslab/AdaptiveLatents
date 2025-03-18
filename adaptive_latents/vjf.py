from adaptive_latents.transformer import DecoupledTransformer
from adaptive_latents.timed_data_source import ArrayWithTime
import numpy as np
import torch
import vjf.online
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

class BaseVJF:
    def __init__(self, *, config=None, latent_d=6, take_U=False):
        self.latent_d = latent_d
        self.take_U = take_U
        config = config or {}
        self.config = self.default_config_dict({'xdim': self.latent_d}) | config
        self.vjf: vjf.online.VJF | None = None
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

        self.vjf = vjf.online.VJF(self.config)

    def generate_cloud(self, rng, q=None, n_points=1000):
        if q is None:
            if self.q is None:
                q = torch.zeros(1, self.latent_d), torch.zeros(1, self.latent_d)
            else:
                q = self.q
        filtering_mu, filtering_logvar = q

        mu_f = filtering_mu[0].detach().cpu().numpy().T
        var_f = filtering_logvar[0].detach().exp().cpu().numpy().T
        Sigma_f = np.eye(filtering_mu.shape[1]) * var_f

        x = multivariate_normal(mu_f.flatten(), Sigma_f).rvs(size=n_points, random_state=rng).astype(np.float32)
        x = torch.from_numpy(x)
        return x

    def step_for_cloud(self, cloud, rng):
        mdl = self.vjf
        cloud += mdl.system.velocity(cloud) + mdl.system.noise.var ** 0.5 * torch.from_numpy(
            rng.normal(size=cloud.shape))
        return cloud

    def get_logprob_for_cloud(self, cloud, point):
        y_var = self.vjf.likelihood.logvar.detach().exp().cpu().numpy().T

        y_tilde = self.vjf.decoder(cloud).detach().cpu().numpy()

        sample_logprobs = [self.diagonal_normal_logpdf(y_est, y_var, point) for y_est in y_tilde]
        logprob = logsumexp(sample_logprobs) - np.log(cloud.shape[0])
        return logprob

    def get_distance_for_cloud(self, cloud, point):
        y_tilde = self.vjf.decoder(cloud).detach().cpu().numpy()
        distance = np.linalg.norm(y_tilde - point, axis=-1).mean()
        return distance

    def get_cloud_at_time_t(self, q, n_points, rng, n_steps):
        with torch.no_grad():
            cloud = self.generate_cloud(q, n_points, rng)
            for _ in range(n_steps):
                cloud = self.step_for_cloud(cloud, rng)
            return cloud

    def observe(self, y_t, u_t):
        self.q, loss = self.vjf.feed((y_t, u_t), self.q)

    @staticmethod
    def diagonal_normal_logpdf(mean, variance, sample):
        mean = mean.flatten()
        variance = variance.flatten()
        sample = sample.flatten()

        assert len(mean) == len(variance) == len(
            sample), f"inconsistent shape: {mean.shape}, {variance.shape}, {sample.shape}"

        logprobs = []
        for i in range(len(sample)):
            x = sample[i]
            m = mean[i]
            v = variance[i]
            logprobs.append(-0.5 * ((x - m) ** 2 / v + np.log(2 * np.pi * v)))
        return sum(logprobs)


class VJF(DecoupledTransformer, BaseVJF):
    base_algorithm = BaseVJF

    def __init__(self, *, config=None, latent_d=6, take_U=False, input_streams=None, output_streams=None, log_level=None):
        if input_streams is None:
            input_streams = {1: 'U'} if take_U else {}
            input_streams = input_streams | {0: 'Y', 2:'dt'}
        DecoupledTransformer.__init__(self=self, input_streams=input_streams, output_streams=output_streams,
                                      log_level=log_level)
        BaseVJF.__init__(self, config=config, latent_d=latent_d, take_U=take_U)
        self.last_seen = {}


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