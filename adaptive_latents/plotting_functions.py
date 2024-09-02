import datetime
import pathlib
import numpy as np
import matplotlib.pylab as plt
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegFileWriter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import adaptive_latents
    from adaptive_latents import BWRun, CONFIG


class AnimationManager:
    def __init__(self, filename=None, outdir=None, n_rows=1, n_cols=1, fps=20, dpi=100, extension="mp4", figsize=(10,10), projection='rectilinear', make_axs=True):
        outdir = outdir or CONFIG['plot_save_path']
        if filename is None:
            time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            filename = f'movie_{time_string}'
        self.outfile = pathlib.Path(outdir).resolve() / f"{filename}.{extension}"
        self.movie_writer = FFMpegFileWriter(fps=fps)
        if make_axs:
            # TODO: rename ax to axs
            self.fig, self.axs = plt.subplots(n_rows, n_cols, figsize=figsize, layout='tight', squeeze=False, subplot_kw={'projection': projection})
        else:
            self.fig = plt.figure(figsize=figsize, layout='tight')
        self.movie_writer.setup(self.fig, self.outfile, dpi=dpi)
        self.seen_frames = 0
        self.finished = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seen_frames:
            self.finish()

    def finish(self):
        if not self.finished:
            self.movie_writer.finish()
            self.finished = True

    def grab_frame(self):
        self.movie_writer.grab_frame()
        self.seen_frames += 1




def _limits(data):
    low = min(data)
    high = max(data)
    return (high-low) / 2, (high+low) / 2


def use_bigger_lims_from_data(ax, data, padding_proportion=0.05):
    x_span, x_center = _limits(data[:, 0])
    x_span *= (1 + padding_proportion/2)
    current_xlim = ax.get_xlim()
    ax.set_xlim([min(current_xlim[0], x_center - x_span), max(current_xlim[1], x_center + x_span)])

    y_span, y_center = _limits(data[:, 1])
    y_span *= (1 + padding_proportion/2)
    current_ylim = ax.get_ylim()
    ax.set_ylim([min(current_ylim[0], y_center - y_span), max(current_ylim[1], y_center + y_span)])


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


def show_data_2d(ax, data, bw, n=10):
    ax.cla()
    ax.scatter(data[:, 0], data[:, 1], s=5, color='#004cff', alpha=np.power(1 - bw.eps, np.arange(data.shape[0], 0, -1)))

    start = max(data.shape[0] - n, 0)
    ax.plot(data[start:, 0], data[start:, 1], linewidth=3, color='#004cff', alpha=.5)
    use_bigger_lims_from_data(ax, data)


def _mean_distance(data, shift=1):
    x = data - data.mean(axis=0)
    T = x.shape[0]
    differences = x[0:T - shift] - x[shift:T]
    distances = np.linalg.norm(differences, axis=1)
    return distances.mean()


def show_data_distance(ax, d, max_step=50):
    """like a distance autocovariance function"""

    old_ylim = ax.get_ylim()
    ax.cla()
    if d.shape[0] > 10:
        shifts = np.arange(0, min(d.shape[0] // 2, max_step))
        distances = [_mean_distance(d, shift) for shift in shifts]
        ax.plot(shifts, distances)
    ax.set_xlim([0, max_step])
    new_ylim = ax.get_ylim()
    ax.set_ylim([0, max(old_ylim[1], new_ylim[1])])
    ax.set_title(f"dataset distances")
    ax.set_xlabel("offset")
    ax.set_ylabel("distance")




def _one_sided_ewma(data, com=100):
    import pandas as pd
    return pd.DataFrame(data=dict(data=data)).ewm(com).mean()["data"]

def compare_bw_runs(bws):
    offset = 1

    for idx, bw in enumerate(bws):
        ...
        # plot prediction
        # plot entropy
        # give text for both



def compare_metrics(brs, offset, colors=None, show_target_times=False, smoothing_scale=50, show_legend=True, show_title=True, red_lines=(), minutes=False, include_behavior=True, include_trendlines=True, red_lines_frames=None, xlim=None):
    colors = ["black"] + [f"C{i}" for i in range(len(brs) - 1)]
    ps = [br.bw.get_parameters() for br in brs]
    keys = set([leaf for tree in ps for leaf in tree.keys()])
    keep_keys = []
    for key in keys:
        values = [d.get(key) for d in ps]
        if not all([values[0] == v for v in values]):
            keep_keys.append(key)
    to_print = []
    for key in keep_keys:
        to_print.append(f"{key}: {[p.get(key) for p in ps]}")

    if hasattr(brs[0].h, 'beh_error'):
        beh_plot_raw_data = brs[0].h.beh_error[offset]
        if beh_plot_raw_data.size == 0 or np.all(np.isnan(beh_plot_raw_data)):
            include_behavior = False
    else:
        include_behavior = False

    fig, ax = plt.subplots(figsize=(14, 5), nrows=2 + include_behavior, ncols=2, sharex='col', layout='tight', gridspec_kw={'width_ratios': [7, 1]})
    fig: plt.Figure
    to_write = [[] for _ in range(ax.shape[0])]
    last_half_times = []

    for idx, br in enumerate(brs):
        br: adaptive_latents.bw_run.BWRun

        predictions = br.h.log_pred_p[offset]
        if show_target_times:
            bw_offset_t = br.h.bw_offset_t[offset]
        else:
            bw_offset_t = br.h.bw_offset_origin_t[offset]
        smoothed_predictions = _one_sided_ewma(predictions, smoothing_scale)

        if minutes:
            bw_offset_t = bw_offset_t / 60

        c = 'black'
        if colors:
            c = colors[idx]

        last_half_times.append(br.get_last_half_time(offset))
        metrics = br.get_last_half_metrics(offset)

        ax[0, 0].plot(bw_offset_t, predictions, alpha=0.25, color=c)
        if include_trendlines:
            ax[0, 0].plot(bw_offset_t, smoothed_predictions, color=c, label=br.pickle_file.split("/")[-1].split(".")[0].split("_")[-1])
        ax[0, 0].tick_params(axis='y')
        ax[0, 0].set_ylabel('prediction')
        to_write[0].append((idx, f"{metrics['log_pred_p']:.3f}", dict(color=c)))

        entropy = br.h.entropy[offset]
        smoothed_entropy = _one_sided_ewma(entropy, smoothing_scale)

        c = 'black'
        if colors:
            c = colors[idx]
        ax[1, 0].plot(bw_offset_t, entropy, color=c, alpha=0.25)

        if include_trendlines:
            ax[1, 0].plot(bw_offset_t, smoothed_entropy, color=c)
        max_entropy = np.log2(br.bw.N)
        ax[1, 0].axhline(max_entropy, color='k', linestyle='--')

        ax[1, 0].tick_params(axis='y')
        ax[1, 0].set_ylabel('entropy')
        to_write[1].append((idx, f"{metrics['entropy']:.3f}", dict(color=c)))

        if include_behavior:
            beh_error = np.squeeze(br.h.beh_error[offset]**2)
            c = 'black'
            if colors:
                c = colors[idx]
            if show_target_times:
                beh_t = br.h.reg_offset_t[offset]
            else:
                beh_t = br.h.reg_offset_origin_t[offset]

            ax[-1, 0].plot(beh_t, beh_error, color=c)
            ax[-1, 0].set_ylabel('behavior sq.e.')
            ax[-1, 0].tick_params(axis='y')

            to_write[2].append((idx, " ".join([f"{x :.2f}" for x in metrics['beh_sq_error']]), dict(color=c)))

    for axis in ax[:, 0]:
        data_lim = np.array(axis.dataLim).T.flatten()
        bounds = data_lim
        bounds[:2] = (bounds[:2] - bounds[:2].mean()) * np.array([1.02, 1.2]) + bounds[:2].mean()
        bounds[2:] = (bounds[2:] - bounds[2:].mean()) * np.array([1.05, 1.05]) + bounds[2:].mean()

        axis.axis(bounds)

    for i, l in enumerate(to_write):
        for idx, text, kw in l:
            x, y = .93, .93 - .1*idx
            x, y = ax[i, 0].transLimits.inverted().transform([x, y])
            ax[i, 0].text(x, y, text, clip_on=True, verticalalignment='top', **kw)

    if xlim is not None:
        for axis in ax[:, 0]:
            axis.set_xlim(*xlim)

    for axis in ax[:, 0]:
        axis.format_coord = lambda x, y: 'x={:g}, y={:g}'.format(x, y)

    # Add red lines for frame numbers if provided
    if red_lines_frames is not None:
        print(f"Adding red lines at frame numbers: {red_lines_frames}")  # Debug print
        for frame in red_lines_frames:
            for axis in ax[:, 0]:
                if xlim is None or xlim[0] <= frame <= xlim[1]:
                    axis.axvline(frame, color='red', linestyle='--', alpha=0.7)

    if minutes:
        ax[-1, 0].set_xlabel("time (min)")
    else:
        ax[-1, 0].set_xlabel("time (s)")

    xlim = ax[-1, 0].get_xlim()
    xticks = list(ax[-1, 0].get_xticks())
    xtick_labels = list(ax[-1, 0].get_xticklabels())
    ax[-1, 0].set_xticks(xticks + list(set(last_half_times)))
    ax[-1, 0].set_xticklabels(xtick_labels + [""] * len(set(last_half_times)))
    ax[-1, 0].set_xlim(xlim)



    if show_title:
        ax[0, 0].set_title(" ".join(to_print))
    else:
        print(to_print)
    if show_legend:
        ax[0, 0].legend(loc="lower right")

    gs = ax[0, 1].get_gridspec()
    for a in ax[:, 1]:
        a.remove()
    axbig = fig.add_subplot(gs[:, 1])
    axbig.axis("off")
    to_write = "\n".join([f"{k}: {v}" for k, v in ps[0].items()])
    to_write += f"\n\ntime: {br.runtime_since_init:.1f} s\nend of dataset? {'y' if br.hit_end_of_dataset else 'n'}"
    for note in br.notes:
        to_write += f"\n{note}"
    axbig.text(0, 1, to_write, transform=axbig.transAxes, verticalalignment="top")


