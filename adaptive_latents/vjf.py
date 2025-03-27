import matplotlib.pyplot as plt
import numpy as np
import torch
import vjf.online
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from adaptive_latents.predictor import Predictor
from adaptive_latents.timed_data_source import ArrayWithTime


class BaseVJF:
    def __init__(self, *, config=None, latent_d=6, take_U=False, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        else:
            self.set_torch_seeds(rng)
        self.rng = rng
        self.latent_d = latent_d
        self.take_U = take_U
        config = config or {}
        self.config = self.default_config_dict({'xdim': self.latent_d}) | config
        self._vjf: vjf.online.VJF | None = None
        self.q = None  # center, variance

    @staticmethod
    def set_torch_seeds(rng):
        seed = rng.integers(0, int(1e12), 1)[0]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

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

        self._vjf = vjf.online.VJF(self.config)

    def generate_cloud(self, q=None, n_points=1000):
        if q is None:
            if self.q is None:
                q = torch.zeros(1, self.latent_d), torch.zeros(1, self.latent_d)
            else:
                q = self.q
        filtering_mu, filtering_logvar = q

        mu_f = filtering_mu[0].detach().numpy().T
        var_f = filtering_logvar[0].detach().exp().numpy().T
        Sigma_f = np.eye(filtering_mu.shape[1]) * var_f

        x = multivariate_normal(mu_f.flatten(), Sigma_f).rvs(size=n_points, random_state=self.rng).astype(np.float32)
        x = torch.from_numpy(x)
        return x

    def step_for_cloud(self, cloud):
        mdl = self._vjf
        cloud += mdl.system.velocity(cloud) + mdl.system.noise.var ** 0.5 * torch.from_numpy(
            self.rng.normal(size=cloud.shape))
        return cloud

    def get_logprob_for_cloud(self, cloud, point):
        y_var = self._vjf.likelihood.logvar.detach().exp().numpy().T

        decoded_cloud = self._vjf.decoder(cloud).detach().numpy()

        sample_logprobs = [self.diagonal_normal_logpdf(y_est, y_var, point) for y_est in decoded_cloud]
        logprob = logsumexp(sample_logprobs) - np.log(cloud.shape[0])
        return logprob

    def get_distance_for_cloud(self, cloud, point):
        y_tilde = self._vjf.decoder(cloud).detach().numpy()
        distance = np.linalg.norm(y_tilde - point, axis=-1).mean()
        return distance

    def get_cloud_at_time_t(self, n_steps, n_points=500, q=None):
        if q is None:
            q = self.q

        with torch.no_grad():
            cloud = self.generate_cloud(q, n_points)
            for _ in range(n_steps):
                cloud = self.step_for_cloud(cloud)
            return cloud

    def observe(self, y_t, u_t, grad_kwargs):
        if self._vjf is None:
            self.init_vjf(ydim=y_t.shape[-1], udim=u_t.shape[-1])

        self.q, loss = self._vjf.feed((y_t, u_t), self.q, **grad_kwargs)

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

    def predict(self, n_steps, n_points=None, method='mean'):
        cloud = self.get_cloud_at_time_t(n_steps, n_points=n_points)

        if method == 'mean':
            return self._vjf.decoder(cloud).detach().numpy().mean(axis=0)
        elif method == 'most_likely':
            return self.get_most_likely_decoded_point_from_cloud(cloud)

    def get_most_likely_decoded_point_from_cloud(self, cloud):
        # also see get_logprob_for_cloud
        var = self._vjf.likelihood.logvar.detach().exp().numpy().T

        decoded_points = self._vjf.decoder(cloud).detach()
        S = decoded_points.shape[0]

        pairwise_logprobs = np.zeros((S,S)) * np.nan
        for i in range(len(decoded_points)):
            for j in range(i+1):
                pairwise_logprobs[i,j] = self.diagonal_normal_logpdf(decoded_points[i], var, decoded_points[j])
                pairwise_logprobs[j,i] = pairwise_logprobs[i,j]

        probs = logsumexp(pairwise_logprobs, axis=0) - np.log(S)
        return decoded_points[np.argmax(probs)].numpy()

    def fit(self, y, u):
        if self._vjf is None:
            self.init_vjf(ydim=y.shape[-1], udim=u.shape[-1])

        mu, logvar, losses = self._vjf.filter(y[None, ...], u[None, ...])

        self.q = (mu[-1], logvar[-1])

        mu = mu.detach().numpy()[:, 0, :]
        logvar = logvar.detach().numpy()[:, 0, :]
        losses = torch.tensor(losses).detach().numpy()

        return mu, logvar, losses

    def show_nstep_cloud(self, n_steps, true_data=None, lims=(-20,20)):

        cloud = self.get_cloud_at_time_t(n_steps).detach()
        preds = self._vjf.decoder(cloud).detach().numpy()

        n = 31
        x_edges = np.linspace(*lims, n)
        y_edges = np.linspace(*lims, n)
        x_centers = np.convolve([.5, .5], x_edges, mode='valid')
        y_centers = np.convolve([.5, .5], y_edges, mode='valid')
        log_probs = np.zeros((len(y_centers), len(x_centers)))
        for i, y_i in enumerate(y_centers):
           for j, x_j in enumerate(x_centers):
               log_probs[i, j] = self.get_logprob_for_cloud(cloud=cloud, point=np.array([x_j, y_i] + (preds.shape[1]-2)*[0]))
        i, j = np.unravel_index(np.argmax(log_probs), log_probs.shape)

        fig, axs = plt.subplots(ncols=2, figsize=(10, 4), sharey=True, sharex=True,
                               subplot_kw={'adjustable': 'box', 'aspect': 1})
        if true_data is not None:
            assert true_data.shape[1] == 2
            axs[0].plot(true_data[:, 0], true_data[:, 1])

        axs[0].scatter(preds[:, 0], preds[:, 1], color='C1', s=5, zorder=3)
        axs[0].scatter(preds[:, 0].mean(), preds[:, 1].mean(), color='C2', zorder=3)

        axs[1].pcolormesh(x_edges, y_edges, log_probs, vmin=np.quantile(log_probs.flatten(), .5), vmax=log_probs.max(),
                         cmap='plasma')
        axs[1].scatter(x_centers[j], y_centers[i], color='C2')

        point = self.get_most_likely_decoded_point_from_cloud(cloud)
        axs[1].scatter(point[0], point[1], color='C3')
        return fig, axs

    def __getstate__(self):
        return super().__getstate__() | dict(q=tuple(x.detach() for x in self.q))


class VJF(Predictor, BaseVJF):
    base_algorithm = BaseVJF

    def __init__(self, *, config=None, latent_d=6, rng=None, take_U=False, n_particles_for_prediction=500, check_dt=False, input_streams=None, output_streams=None, log_level=None):
        if input_streams is None:
            input_streams = {1: 'U'} if take_U else {}
            input_streams = input_streams | {0: 'X', 2:'dt'}
        Predictor.__init__(self=self, input_streams=input_streams, output_streams=output_streams, log_level=log_level, check_dt=check_dt)
        BaseVJF.__init__(self, config=config, latent_d=latent_d, take_U=take_U, rng=rng)
        self.n_particles_for_prediction = n_particles_for_prediction
        self.last_seen = {}

    def observe(self, X, stream=None):
        if self.input_streams[stream] in ['X', 'U']:
            self.last_seen[self.input_streams[stream]] = X
            if len(self.last_seen) == (1 + self.take_U):
                y, u = self.get_y_and_u()

                grad_kwargs = {m:self.parameter_fitting for m in ['decoder', 'encoder', 'dynamics', 'noise']}
                BaseVJF.observe(self, y, u, grad_kwargs=grad_kwargs)

    def predict(self, n_steps):
        if self.q is None:
            return np.nan
        return BaseVJF.predict(self, n_steps, n_points=self.n_particles_for_prediction, method='mean')

    def get_state(self):
        if self.q is None:
            return np.nan

        return self.q[0].detach().numpy()

    def get_arbitrary_dynamics_parameter(self):
        if self.q is None:
            return np.nan
        return self._vjf.system.transition.weight.detach().numpy()

    def get_y_and_u(self, y=None, u=None):
        if y is None:
            y = self.last_seen['X']

        if u is None:
            if self.take_U:
                u = self.last_seen['U']
            else:
                u = np.zeros((y.shape[0], 1))

        y = torch.from_numpy(y).float()
        u = torch.from_numpy(u).float()

        return y, u

    def fit(self, y, u=None):
        y, u = self.get_y_and_u(y, u)
        return super().fit(y, u)

    def get_params(self, deep=True):
        return super().get_params(deep=deep) | dict(take_U=self.take_U, latent_d=self.latent_d, config=self.config, rng=self.rng, n_particles_for_prediction=self.n_particles_for_prediction)