import datetime
import pathlib
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
import matplotlib.gridspec as gridspec
import itertools
import adaptive_latents
import numpy as np
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from adaptive_latents import CONFIG


class AnimationManager:
    def __init__(self, filename=None, outdir=None, n_rows=1, n_cols=1, fps=20, dpi=100, filetype="mp4", figsize=(10, 10), projection='rectilinear', make_axs=True, fig=None):
        outdir = outdir or CONFIG['plot_save_path']
        if filename is None:
            time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            filename = f"movie_{time_string}-{str(hash(id(self)))[-3:]}"

        self.outfile = pathlib.Path(outdir).resolve() / f"{filename}.{filetype}"
        Writer = FFMpegWriter
        if filetype == 'gif':
            Writer = PillowWriter
        self.movie_writer = Writer(fps=fps)
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

def plot_history_with_tail(ax, data, tail_length=10, dim_1=0, dim_2=1,):
    ax.cla()

    ax.scatter(data[:,dim_1], data[:,dim_2], s=5, alpha=.1, c='C0', edgecolors='none')

    linewidth = 2
    s = 10
    ax.plot(data[-tail_length:, dim_1], data[-tail_length:, dim_2], color='white', linewidth=linewidth * 1.5)
    ax.scatter(data[-1, dim_1], data[-1, dim_2], s=s * 1.5, color='white')
    ax.plot(data[-tail_length:, dim_1], data[-tail_length:, dim_2], color='C0', linewidth=linewidth)
    ax.scatter(data[-1,dim_1], data[-1,dim_2], s=s, zorder=3)


def plot_bw_pipeline(bws, behavior_dict_list=None, t_in_samples=False):
    import matplotlib.pyplot as plt
    def _one_sided_ewma(data, com=100):
        import pandas as pd
        return pd.DataFrame(data=dict(data=data)).ewm(com).mean()["data"]

    def plot_with_trendline(ax, times, data, color, com=100):
        ax.plot(times, data, alpha=.25, color=color)
        smoothed_data = _one_sided_ewma(data, com, )
        ax.plot(times, smoothed_data, color=color)

    bws: [adaptive_latents.Bubblewrap]
    assert len(bws) == 1
    for bw in bws:
        assert bw.log_level > 0


    fig, axs = plt.subplots(figsize=(14, 5 + 2 * len(behavior_dict_list)), nrows=2 + len(behavior_dict_list), ncols=2, sharex='col', layout='tight',
                            gridspec_kw={'width_ratios': [7, 1]})

    common_time_start = max([min(bw.log['t']) for bw in bws])
    common_time_end = min([max(bw.log['t']) for bw in bws])
    halfway_time = (common_time_start + common_time_end) / 2

    to_write = [[] for _ in range(axs.shape[0])]
    colors = ['C0'] + ['k'] * (len(bws) - 1)

    for idx, bw in enumerate(bws): # only happens once
        color = colors[idx]

        # plot prediction
        t = np.array(bw.log['log_pred_p_origin_t'])
        t_to_plot = t
        if t_in_samples:
            t_to_plot = t / bw.dt
        to_plot = np.array(bw.log['log_pred_p'])
        plot_with_trendline(axs[0, 0], t_to_plot, to_plot, color)
        last_half_mean = to_plot[(halfway_time < t) & (t < common_time_end)].mean()
        to_write[0].append((idx, f'{last_half_mean:.2f}', {'color': color}))
        axs[0, 0].set_ylabel('log pred. p')

        # plot entropy
        t = np.array(bw.log['t'])
        t_to_plot = t
        if t_in_samples:
            t_to_plot = t / bw.dt
        to_plot = np.array(bw.log['entropy'])
        plot_with_trendline(axs[1, 0], t_to_plot, to_plot, color)
        last_half_mean = to_plot[(halfway_time < t) & (t < common_time_end)].mean()
        to_write[1].append((idx, f'{last_half_mean:.2f}', {'color': color}))
        axs[1, 0].set_ylabel('entropy')

        max_entropy = np.log2(bw.N)
        axs[1, 0].axhline(max_entropy, color='k', linestyle='--')

    # plot behavior
    for idx, behavior_dict in enumerate(behavior_dict_list):
        from adaptive_latents.utils import resample_matched_timeseries
        ax_n = 2 + idx

        t = behavior_dict['predicted_behavior_t']
        targets = resample_matched_timeseries(
            behavior_dict['true_behavior'],
            behavior_dict['true_behavior_t'],
            t
        )
        estimates = behavior_dict['predicted_behavior']

        test_s = t > (t[0] + t[-1]) / 2

        correlations = [np.corrcoef(estimates[test_s, i], targets[test_s, i])[0, 1] for i in range(estimates.shape[1])]
        corr_str = '\n'.join([f'{r:.2f}' for r in correlations] )
        to_write[ax_n].append((idx, corr_str, {'fontsize': 'large'}))

        t_to_plot = t
        if t_in_samples:
            t_to_plot = t / np.median(np.diff(t))
        for i in range(estimates.shape[1]):
            axs[ax_n,0].plot(t_to_plot, targets[:, i], color=f'C{i}')
            axs[ax_n,0].plot(t_to_plot, estimates[:, i], color=f'C{i}', alpha=.5)
        axs[ax_n,0].set_xlabel("time")
        axs[ax_n,0].set_ylabel(behavior_dict['label'])


    # this sets the axis bounds for the text
    for axis in axs[:, 0]:
        data_lim = np.array(axis.dataLim).T.flatten()
        bounds = data_lim
        bounds[:2] = (bounds[:2] - bounds[:2].mean()) * np.array([1.02, 1.2]) + bounds[:2].mean()
        bounds[2:] = (bounds[2:] - bounds[2:].mean()) * np.array([1.05, 1.05]) + bounds[2:].mean()
        axis.axis(bounds)
        axis.format_coord = lambda x, y: 'x={:g}, y={:g}'.format(x, y)

    # this prints the last-half means
    for i, l in enumerate(to_write):
        for idx, text, kw in l:
            x, y = .92, .93 - .1 * idx
            x, y = axs[i, 0].transLimits.inverted().transform([x, y])
            axs[i, 0].text(x, y, text, clip_on=True, verticalalignment='top', **kw)

    # this creates the axis for the parameters
    gs = axs[0, 1].get_gridspec()
    for a in axs[:, 1]:
        a.remove()
    axbig = fig.add_subplot(gs[:, 1])
    axbig.axis("off")

    # this generates and prints the parameters
    params_per_bw_list = [bw.get_params() for bw in bws]
    super_param_dict = {}
    for key in params_per_bw_list[0].keys():
        values = [p[key] for p in params_per_bw_list]
        if len(set(values)) == 1:
            values = values[0]
            if key in {'input_streams', 'output_streams', 'log_level'}:
                continue
        super_param_dict[key] = values
    to_write = "\n".join(f"{k}: {v}" for k, v in super_param_dict.items())
    axbig.text(0, 1, to_write, transform=axbig.transAxes, verticalalignment="top")



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
        # TODO: this is pretty inflexible, remove refrences to `d.` with assumptions
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
        plot_history_with_tail(ax, latents)
        lines = ax.plot([latents[-1, 0], latent_predictions[-1, 0]], [latents[-1, 1], latent_predictions[-1, 1]],
                '--', color=self.pred_color,)
        ax.scatter(latents[-1, 0], latents[-1, 1], s=lines[0].get_linewidth(), color=self.pred_color, zorder=4)
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
        ax.plot(beh_t, beh)
        use_bigger_lims(ax, old_lims=old_lims, x=False)

        for i in range(beh_to_predict.a.shape[2]):
            ax.plot([beh_to_predict.t[idx], prediction_ts[-1]],
                    [beh_to_predict.a[idx, 0, i], beh_predictions[-1,i]], '--', color=f'C{i}', alpha=.5)

        beh_dt = np.median(np.diff(prediction_ts))
        n_columns = np.floor(self.tail_length / beh_dt).astype(int)
        idx = np.nonzero(prediction_ts > current_t)[0][0]
        s = slice(idx - n_columns, idx)
        for i in range(beh_to_predict.a.shape[2]):
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
