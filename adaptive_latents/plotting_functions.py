import datetime
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
import matplotlib.gridspec as gridspec
import functools
import numpy as np
import pathlib
import warnings
from adaptive_latents import CONFIG
from adaptive_latents.timed_data_source import ArrayWithTime
from IPython import display


class AnimationManager:
    """
    Examples
    --------
    >>> tmp_path = getfixture('tmp_path')
    >>> with AnimationManager(outdir=tmp_path) as am:
    ...     for i in range(2):
    ...         for ax in am.axs.flatten():
    ...             ax.cla()
    ...         # animation things would go here
    ...         am.grab_frame()
    ...     fpath = am.outfile
    >>> assert fpath.is_file()
    """
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
    """
    Examples
    --------
    >>> fig1, ax1 = plt.subplots()
    >>> _ = ax1.plot([0, 1], [0, 1])
    >>> fig2, ax2 = plt.subplots()
    >>> _ = ax2.plot([0, .1], [0, .1])
    >>> use_bigger_lims(ax2, old_lims=ax1.axis())
    """
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
    """
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> X = np.random.normal(size=(100,2))
    >>> X = ArrayWithTime.from_notime(X)
    >>> plot_history_with_tail(ax, data=X, current_t=75, tail_length=4, scatter_alpha=1)
    """
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
        >>> count = 0
        >>> for v in o.suggest_values(0,6.28):
        ...     time.sleep(.01)
        ...     o.register_result(v, {'beh':{'corr': [np.sin(v), np.cos(v)], 'nrmse': -np.sin(v)}, 'joint':{'corr': [np.cos(v), np.cos(v*2)], 'nrmse': np.cos(v)}})
        ...     break  # ususally this would keep going in a notebook until a keyboard interrupt
        <BLANKLINE>
        ... Figure(640x480)
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


def plot_flow_fields(dim_reduced_data, x_direction=0, y_direction=1, grid_n=13, scatter_alpha=0, normalize_method=None, fig=None):
    """
    Examples
    --------
    >>> X = np.random.normal(size=(100,2))
    >>> plot_flow_fields({'random points': X}, normalize_method='squares', grid_n=20)
    """
    assert normalize_method in {None, 'none', 'diffs', 'hcubes', 'squares'}
    if fig is None:
        fig, axs = plt.subplots(nrows=1, ncols=len(dim_reduced_data), squeeze=False, layout='tight', figsize=(12,4))
    else:
        axs = fig.axes

    for idx, (name, latents) in enumerate(dim_reduced_data.items()):
        e1, e2 = np.zeros(latents.shape[1]), np.zeros(latents.shape[1])
        e1[x_direction] = 1
        e2[y_direction] = 1

        ax: plt.Axes = axs[0, idx]
        ax.scatter(latents @ e1, latents @ e2, s=5, alpha=scatter_alpha)
        x1, x2, y1, y2 = ax.axis()
        x_points = np.linspace(x1, x2, grid_n)
        y_points = np.linspace(y1, y2, grid_n)

        d_latents = np.diff(latents, axis=0)
        if normalize_method == 'diffs':
            d_latents = d_latents / np.linalg.norm(d_latents, axis=1)[:, np.newaxis]


        origins = []
        arrows = []
        n_points = []
        for i in range(len(x_points) - 1):
            for j in range(len(y_points) - 1):
                proj_1 = (latents[:-1] @ e1)
                proj_2 = (latents[:-1] @ e2)
                # s stands for slice
                s = (
                        (x_points[i] <= proj_1) & (proj_1 < x_points[i + 1])
                        &
                        (y_points[j] <= proj_2) & (proj_2 < y_points[j + 1])
                )
                if s.sum():
                    arrow = np.nanmean(d_latents[s],axis=0)
                    if normalize_method == 'hcubes':
                        arrow = arrow / np.linalg.norm(arrow)
                    arrows.append(arrow)
                    origins.append([np.nanmean(x_points[i:i + 2]), np.nanmean(y_points[j:j + 2])])
                    n_points.append(s.sum())
        origins, arrows, n_points = np.array(origins), np.array(arrows), np.array(n_points)
        arrows = np.array([arrows @ e1, arrows @ e2]).T
        if normalize_method == 'squares':
            arrows = arrows / np.linalg.norm(arrows, axis=1)[:, np.newaxis]

        ax.quiver(origins[:, 0], origins[:, 1], arrows[:,0], arrows[:,1], scale=1 / 20, units='dots', color='red')

        ax.axis('equal')
        ax.axis('off')
