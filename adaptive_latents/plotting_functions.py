import datetime
import pathlib
import matplotlib.pylab as plt
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegFileWriter
import matplotlib.gridspec as gridspec
from adaptive_latents import CONFIG
import numpy as np
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from adaptive_latents import CONFIG


class AnimationManager:
    def __init__(self, filename=None, outdir=None, n_rows=1, n_cols=1, fps=20, dpi=100, extension="mp4", figsize=(10,10), projection='rectilinear', make_axs=True, fig=None):
        outdir = outdir or CONFIG['plot_save_path']
        if filename is None:
            time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            filename = f'movie_{time_string}'
        self.outfile = pathlib.Path(outdir).resolve() / f"{filename}.{extension}"
        self.movie_writer = FFMpegFileWriter(fps=fps)
        if fig is None:
            if make_axs:
                self.fig, self.axs = plt.subplots(n_rows, n_cols, figsize=figsize, layout='tight', squeeze=False, subplot_kw={'projection': projection})
            else:
                self.fig = plt.figure(figsize=figsize, layout='tight')
        else:
            self.fig = fig
        self.movie_writer.setup(self.fig, self.outfile, dpi=dpi)
        self.seen_frames = 0
        self.finished = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seen_frames:
            self.finish()
        else:
            warnings.warn('closed without any frame grabs')

    def finish(self):
        if not self.finished:
            self.movie_writer.finish()
            self.finished = True

    def grab_frame(self):
        self.movie_writer.grab_frame()
        self.seen_frames += 1


def use_bigger_lims(ax, old_lims=None, y=True, x=True):
    new_lims = ax.axis()
    if old_lims is None:
        old_lims = new_lims

    future_lims = [min(old_lims[0], new_lims[0]), max(old_lims[1], new_lims[1]), min(old_lims[2], new_lims[2]), max(old_lims[3], new_lims[3])]
    if not y:
        future_lims[2:] = new_lims[2:]

    if not x:
        future_lims[:2] = new_lims[:2]
    ax.axis(future_lims)

def plot_history_with_tail(ax, data, tail_length=10):
    ax.cla()

    ax.scatter(data[:,0], data[:,1], s=5, alpha=.1, c='C0', edgecolors='none')

    linewidth = 2
    s = 10
    ax.plot(data[-tail_length:, 0], data[-tail_length:, 1], color='white', linewidth=linewidth * 1.5)
    ax.scatter(data[-1, 0], data[-1, 1], s=s * 1.5, color='white')
    ax.plot(data[-tail_length:, 0], data[-tail_length:, 1], color='C0', linewidth=linewidth)
    ax.scatter(data[-1,0], data[-1,1], s=s, zorder=3)


class PredictionVideo:
    def __init__(self, d, initialization_phase_cutoff_t=20, tail_length=2, fps=None):
        self.d = d
        self.tail_length = tail_length
        self.pred_color = '#b337a4'
        self.initialization_phase_cutoff_t = initialization_phase_cutoff_t
        fps = fps or 10

        self.fig = plt.figure(constrained_layout=False, dpi=200, figsize=(15, 5))
        spec = gridspec.GridSpec(ncols=3, nrows=2, figure=self.fig, height_ratios=[19, 1], hspace=0.01)
        self.neural_data_ax = self.fig.add_subplot(spec[0, 0])
        self.beh_data_ax = self.fig.add_subplot(spec[1, 0])
        self.joint_latent_ax = self.fig.add_subplot(spec[:, 1])
        self.prediction_ax = self.fig.add_subplot(spec[:, 2])

        self.am = AnimationManager(fig=self.fig, fps=fps, dpi=self.fig.dpi)

    def plot_for_video_t(self, current_t, latents, latent_ts, latent_predictions, beh_predictions, prediction_ts):
        latents = np.squeeze(latents)
        latent_ts = np.squeeze(latent_ts)
        s = latent_ts > self.initialization_phase_cutoff_t
        latents, latent_ts = latents[s], latent_ts[s]

        latent_predictions = np.squeeze(latent_predictions)
        beh_predictions = np.squeeze(beh_predictions)
        prediction_ts = np.squeeze(prediction_ts)
        s = prediction_ts > self.initialization_phase_cutoff_t
        latent_predictions, beh_predictions, prediction_ts = latent_predictions[s], beh_predictions[s], prediction_ts[s]
        if not s.sum():
            return

        for array in [latents, latent_predictions, beh_predictions]:
            assert len(array.shape) == 2


        ax = self.neural_data_ax
        ax.cla()
        n_columns = np.floor(self.tail_length / self.d.bin_width).astype(int)
        idx = np.nonzero(~(self.d.neural_data.t < current_t))[0][0]
        ax.imshow(self.d.neural_data.a[idx - n_columns:idx, 0, :].T, aspect='auto', interpolation='none',
                  extent=[current_t - self.tail_length, current_t, self.d.neural_data.a.shape[2], 0])

        ax = self.beh_data_ax
        ax.cla()
        beh_dt = np.median(np.diff(self.d.behavioral_data.t))
        n_columns = np.floor(self.tail_length / beh_dt).astype(int)
        idx = np.nonzero(~(self.d.behavioral_data.t < current_t))[0][0]
        ax.imshow(self.d.behavioral_data.a[idx - n_columns:idx, 0, :].T, aspect='auto', interpolation='none',
                  extent=[current_t - self.tail_length, current_t, self.d.neural_data.a.shape[2], 0])

        ax = self.joint_latent_ax
        old_lims = ax.axis()
        plot_history_with_tail(ax, latents)
        ax.plot([latents[-1, 0], latent_predictions[-1, 0]], [latents[-1, 1], latent_predictions[-1, 1]],
                '--', color=self.pred_color)
        use_bigger_lims(ax, old_lims=old_lims)


        ax = self.prediction_ax
        old_lims = ax.axis()
        ax.cla()
        beh_dt = np.median(np.diff(self.d.behavioral_data.t))
        n_columns = np.floor(self.tail_length / beh_dt).astype(int)
        idx = np.nonzero(~(self.d.behavioral_data.t < current_t))[0][0]
        s = slice(idx - n_columns, idx)
        beh = self.d.behavioral_data.a[s, 0, :]
        beh_t = self.d.behavioral_data.t[s]
        ax.plot(beh_t, beh)
        use_bigger_lims(ax, old_lims=old_lims, x=False)

        for i in range(3):
            ax.plot([self.d.behavioral_data.t[idx], prediction_ts[-1]],
                    [self.d.behavioral_data.a[idx, 0, i], beh_predictions[-1,i]], '--', color=f'C{i}', alpha=.5)

        beh_dt = np.median(np.diff(prediction_ts))
        n_columns = np.floor(self.tail_length / beh_dt).astype(int)
        idx = np.nonzero(~(prediction_ts < current_t))[0][0]
        s = slice(idx - n_columns, idx)
        for i in range(3):
            ax.plot(prediction_ts[s], beh_predictions[s,i], color=f'C{i}', alpha=.25)

            for ax in self.fig.get_axes():
                ax.axis('off')

        self.am.grab_frame()

    @classmethod
    def example_usage(cls, d=None):
        # todo: delete this? move it to tests?
        from adaptive_latents import datasets, Pipeline, KernelSmoother, Concatenator, proSVD, Bubblewrap, VanillaOnlineRegressor, NumpyTimedDataSource
        from tqdm.notebook import tqdm

        if d is None:
            d = datasets.Zong22Dataset()

        p1 = Pipeline([
            KernelSmoother(tau=0.2/d.bin_width, input_streams={0:'X'}, output_streams={0:0}),
            Concatenator(input_streams={0: 0, 1: 1}, output_streams={0:2, 1:2, 'skip':-1}),
            pro:=proSVD(k=6, input_streams={2:'X'}, output_streams={2:2}),
        ])

        p2 = Pipeline([
            bw:=Bubblewrap(
                num=100,
                eps=1e-3,
                step=1,
                num_grad_q=3,
                input_streams={2:'X'},
                output_streams={2:2},
            ),
            reg:=VanillaOnlineRegressor(
                input_streams={2:'X', 3:'Y'},
                output_streams={2:2},
            ),
        ])


        streams = []
        streams.append( (d.neural_data,     0) )
        streams.append( (d.behavioral_data, 1) )
        streams.append( (d.behavioral_data, 3) )

        video_dt = 0.1
        video_ts = np.arange(100)*video_dt + 50
        streams.append( (NumpyTimedDataSource(np.nan * video_ts, video_ts), 'video') )


        vid = cls(d, video_dt)
        pbar = tqdm(total=video_ts[-1])
        latents = []
        latent_ts = []

        latent_predictions = []
        beh_predictions = []
        prediction_ts = []

        with vid.am:
            for output, stream in Pipeline().streaming_run_on(streams, return_output_stream=True):
                # dim reduction part of pipeline
                output, stream = p1.partial_fit_transform(output, stream, return_output_stream=True)
                if stream == 2 and np.isfinite(output).all() and output.t > 20:
                    latents.append(output)
                    latent_ts.append(output.t)

                # prediction part of pipeline
                output, stream = p2.partial_fit_transform(output, stream, return_output_stream=True)
                if stream == 2 and np.isfinite(output).all() and output.t > 20:
                    prediction_t = output.t + bw.dt * 1
                    alpha_pred = bw.get_alpha_at_t(prediction_t)
                    latent_prediction = np.array(bw.mu[np.argmax(alpha_pred)])
                    beh_prediction = reg.predict(alpha_pred)

                    latent_predictions.append(latent_prediction)
                    beh_predictions.append(beh_prediction)
                    prediction_ts.append(prediction_t)

                if stream == 'video':
                    vid.plot_for_video_t(output.t, latents, latent_predictions, beh_predictions, prediction_ts)
                    if output.t == video_ts[-1]:
                        break

                pbar.update(output.t - pbar.n)
