import datetime
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
import matplotlib.gridspec as gridspec
import functools
import numpy as np
import pathlib
import warnings
from adaptive_latents import CONFIG
from IPython import display


class AnimationManager:
    def __init__(self, filename=None, outdir=None, n_rows=1, n_cols=1, fps=20, dpi=100, filetype="webm", figsize=(10, 10), projection='rectilinear', make_axs=True, fig=None):
        if outdir is not None:
            outdir = pathlib.Path(outdir)
        else:
            outdir = CONFIG.plot_save_path
        outdir.parent.mkdir(exist_ok=True, parents=True)

        if filename is None:
            time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            filename = f"movie_{time_string}-{str(hash(id(self)))[-3:]}"

        self.filetype = filetype
        self.outfile = pathlib.Path(outdir).resolve() / f"{filename}.{filetype}"
        Writer = FFMpegWriter
        if filetype == 'gif':
            Writer = PillowWriter
        if filetype == 'webm':
            Writer = functools.partial(FFMpegWriter, codec='libvpx-vp9')

        self.movie_writer = Writer(fps=fps, bitrate=-1)
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

    def display_video(self, embed=False, width=None):
        if self.filetype == 'gif':
            display.display(display.Image(self.outfile, embed=embed, width=width))
        else:
            display.display(display.Video(self.outfile, embed=embed, width=width))


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


def plot_history_with_tail(ax, data, current_t, tail_length=1, scatter_all=True, dim_1=0, dim_2=1, hist_bins=None, invisible=False, scatter_alpha=.1, scatter_s=5):
    ax.cla()

    s = np.ones_like(data.t).astype(bool)
    if scatter_all:
        s = data.t <= current_t
    if hist_bins is None:
        ax.scatter(data[s,dim_1], data[s,dim_2], s=scatter_s, c='gray', edgecolors='none', alpha= 0 if invisible else scatter_alpha)
        back_color = 'white'
        forward_color = 'C0'
    else:
        s = s & np.isfinite(data).all(axis=1)
        ax.hist2d(data[s,dim_1], data[s,dim_2], bins=hist_bins)
        back_color = 'black'
        forward_color = 'white'


    linewidth = 2
    size = 10
    s = (current_t - tail_length < data.t) & (data.t <= current_t)
    ax.plot(data[s, dim_1], data[s, dim_2], color=back_color, linewidth=linewidth * 1.5, alpha= 0 if invisible else 1)
    ax.scatter(data[s, dim_1][-1], data[s, dim_2][-1], s=size * 1.5, color=back_color, alpha= 0 if invisible else 1)
    ax.plot(data[s, dim_1], data[s, dim_2], color=forward_color, linewidth=linewidth, alpha= 0 if invisible else 1)
    ax.scatter(data[s,dim_1][-1], data[s,dim_2][-1], color=forward_color, s=size, zorder=3, alpha= 0 if invisible else 1)
    ax.axis('off')


class UpdatingOptimizationGraph:
    def __init__(self, metrics=None, targets=None, low_is_good_metrics=('nrmse',)):
        """
        Examples
        ----------
        >>> import time
        >>> o = UpdatingOptimizationGraph()
        >>> for v in o.suggest_values(0,6.28):
        >>>     time.sleep(1)
        >>>     o.register_result(v, {'beh':{'corr': [np.sin(v), np.cos(v)], 'nrmse': -np.sin(v)}, 'joint':{'corr': [np.cos(v), np.cos(v*2)], 'nrmse': np.cos(v)}})
        """
        self.fig, self.axs = None, None
        self.low_is_good_metrics = low_is_good_metrics
        self.tried_values = []
        self.results = []
        self.metrics = metrics
        self.targets = targets

    def suggest_values(self, *args, max_n_samples=100):
        while max_n_samples is None or len(self.tried_values) < max_n_samples:
            yield self.binary_search_next_sample(*args, tried_values=self.tried_values)

    def update_plot(self):
        if self.fig is None:
            self.fig, self.axs = plt.subplots(nrows=len(self.targets), ncols=len(self.metrics), squeeze=False)
        for idx, target_str in enumerate(self.targets):
            for jdx, metric_str in enumerate(self.metrics):
                metric = np.array([result[target_str][metric_str] for result in self.results])
                metric = np.atleast_2d(metric.T).T
                self.axs[idx,jdx].cla()
                self.axs[idx,jdx].plot(self.tried_values, metric)

                summaries = metric.sum(axis=1)
                if metric_str in self.low_is_good_metrics:
                    summaries = -summaries
                best_tried = self.tried_values[np.argmax(summaries)]
                self.axs[idx, jdx].axvline(best_tried, color='k', alpha=.5)
                self.axs[idx, jdx].text(.99, .99, f'{best_tried:.3f}', ha='right', va='top', transform=self.axs[idx, jdx].transAxes)

                if idx == 0:
                    self.axs[idx, jdx].set_title(metric_str)
                    self.axs[idx, jdx].set_xticklabels([])

                if idx != len(self.targets) - 1:
                    self.axs[idx, jdx].set_xticks(self.tried_values)
                    self.axs[idx, jdx].set_xticklabels([])

                if jdx == 0:
                    self.axs[idx, jdx].set_ylabel(target_str)

        display.clear_output()
        display.display(self.fig)

    def register_result(self, value, result):
        if isinstance(result, tuple) and hasattr(result, '_asdict'):
            result = result._asdict()
        for k, v in result.items():
            if isinstance(v, tuple) and hasattr(v, '_asdict'):
                result[k] = v._asdict()

        if self.metrics is None or self.targets is None:
            self.targets = list(result.keys())
            self.metrics = list(result[self.targets[0]].keys())

        self.tried_values.append(value)
        self.results.append(result)

        self.tried_values, self.results = map(list, list(zip(*sorted(zip(self.tried_values, self.results)))))

        self.update_plot()

    @staticmethod
    def binary_search_next_sample(*args, tried_values=()):
        # usual args are min, max
        tried = list(tried_values)

        for new_x in args:
            if new_x not in tried:
                return new_x

        tried = sorted(tried)
        idx = np.argmax(np.diff(tried))
        return (tried[idx] + tried[idx + 1]) / 2



class PredictionVideo:
    def __init__(self, d, tail_length=2, fps=None, filetype='mp4'):
        self.d = d
        self.tail_length = tail_length
        self.pred_color = '#b337a4'
        fps = fps or 10

        self.fig = plt.figure(constrained_layout=False, dpi=200, figsize=(15, 5))
        spec = gridspec.GridSpec(ncols=3, nrows=2, figure=self.fig, height_ratios=[19, 1], hspace=0.01)
        self.neural_data_ax = self.fig.add_subplot(spec[0, 0])
        self.beh_data_ax = self.fig.add_subplot(spec[1, 0])
        self.joint_latent_ax = self.fig.add_subplot(spec[:, 1])
        self.prediction_ax = self.fig.add_subplot(spec[:, 2])

        self.am = AnimationManager(fig=self.fig, fps=fps, dpi=self.fig.dpi, filetype=filetype)

    def plot_for_video_t(self, current_t, latents, latent_ts, latent_predictions, beh_predictions, prediction_ts, streams):
        neural_ds, s = streams[0]
        assert s == 0
        beh_input, s = streams[1]
        assert s == 1
        beh_to_predict, s = streams[2]
        assert s == 3

        latents = np.squeeze(latents)
        latent_ts = np.squeeze(latent_ts)

        latent_predictions = np.squeeze(latent_predictions)
        beh_predictions = np.squeeze(beh_predictions)
        prediction_ts = np.squeeze(prediction_ts)

        for array in [latents, latent_predictions, beh_predictions]:
            assert len(array.shape) == 2


        ax = self.neural_data_ax
        ax.cla()
        bin_width = np.median(np.diff(neural_ds.t))
        assert np.isclose(bin_width, self.d.bin_width)
        n_columns = np.floor(self.tail_length / bin_width).astype(int)
        idx = np.nonzero(~(neural_ds.t < current_t))[0][0]
        ax.imshow(neural_ds.a[idx - n_columns:idx, 0, :].T, aspect='auto', interpolation='none',
                  extent=[current_t - self.tail_length, current_t, neural_ds.a.shape[2], 0])

        ax = self.beh_data_ax
        ax.cla()
        beh_dt = np.median(np.diff(beh_input.t))
        n_columns = np.floor(self.tail_length / beh_dt).astype(int)
        idx = np.nonzero(~(beh_input.t < current_t))[0][0]
        ax.imshow(beh_input.a[idx - n_columns:idx, 0, :].T, aspect='auto', interpolation='none',
                  extent=[current_t - self.tail_length, current_t, beh_input.a.shape[2], 0])

        ax = self.joint_latent_ax
        old_lims = ax.axis()
        idx = np.nonzero(~(latents.t < current_t))[0][0]
        
        plot_history_with_tail(ax, latents, current_t, tail_length=.5)

        idx2 = np.nonzero(~(latent_predictions.t < current_t))[0][0]
        lines = ax.plot([latents[idx-1, 0], latent_predictions[idx2, 0]], [latents[idx-1, 1], latent_predictions[idx2, 1]], '--', color=self.pred_color,)
        ax.scatter(latent_predictions[idx2, 0], latent_predictions[idx2, 1], 
                   s=lines[0].get_linewidth(), 
                   color=self.pred_color, zorder=4)
        ax.axis('equal')
        use_bigger_lims(ax, old_lims=old_lims)


        ax = self.prediction_ax
        old_lims = ax.axis()
        ax.cla()
        beh_dt = np.median(np.diff(beh_to_predict.t))
        n_columns = np.floor(self.tail_length / beh_dt).astype(int)
        idx = np.nonzero(~(beh_to_predict.t < current_t))[0][0]
        s = slice(idx - n_columns, idx)
        beh = beh_to_predict.a[s, 0, :]
        beh_t = beh_to_predict.t[s]
        for i in range(beh_to_predict.a.shape[2]):
            ax.plot(beh_t, beh[:,i], color=f'C{i+1}')
        use_bigger_lims(ax, old_lims=old_lims, x=False)

        # idx2 = np.nonzero(~(beh_predictions.t < current_t))[0][0] + 1
        # for i in range(beh_to_predict.a.shape[2]):
        #     ax.plot([beh_to_predict.t[idx], prediction_ts[idx2]],
        #             [beh_to_predict.a[idx, 0, i], beh_predictions[idx2,i]], '--', color=f'C{i+1}', alpha=.5)

        beh_dt = np.median(np.diff(prediction_ts))
        n_columns = np.floor(self.tail_length / beh_dt).astype(int)
        idx = np.nonzero(prediction_ts > current_t)[0][0]
        s = slice(idx - n_columns, idx)
        for i in range(beh_to_predict.a.shape[2]):
            ax.plot(prediction_ts[s], beh_predictions[s,i], color=f'C{i+1}', alpha=.25)

            for ax in self.fig.get_axes():
                ax.axis('off')

        self.am.grab_frame()

    @classmethod
    def example_usage(cls, d=None):
        # todo: delete this? move it to tests?
        from adaptive_latents import datasets, Pipeline, KernelSmoother, Concatenator, proSVD, Bubblewrap, VanillaOnlineRegressor, ArrayWithTime
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
        streams.append((ArrayWithTime(np.nan * video_ts, video_ts), 'video'))


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
