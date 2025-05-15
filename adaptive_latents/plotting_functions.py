import datetime
import functools
import itertools
import pathlib
import warnings

import numpy as np
from IPython import display
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

import adaptive_latents
from adaptive_latents import CONFIG
from adaptive_latents.timed_data_source import ArrayWithTime
from adaptive_latents.utils import resample_matched_timeseries


class AnimationManager:
    """
    Examples
    --------
    >>> tmp_path = getfixture('tmp_path')  # this is mostly for the doctesting framework
    >>> with AnimationManager(outdir=tmp_path) as am:
    ...     for i in range(2):
    ...         for ax in am.axs.flatten():
    ...             ax.cla()
    ...         # animation things would go here
    ...         am.grab_frame()
    ...     fpath = am.outfile
    >>> assert fpath.is_file()
    """
    def __init__(self, filename_stem=None, outdir=None, n_rows=1, n_cols=1, fps=20, dpi=100, filetype="mp4", figsize=(10, 10), projection='rectilinear', make_axs=True, fig=None):
        if outdir is not None:
            outdir = pathlib.Path(outdir)
        else:
            outdir = CONFIG.plot_save_path
        outdir.parent.mkdir(exist_ok=True, parents=True)

        if filename_stem is None:
            time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            filename_stem = f"movie_{time_string}-{str(hash(id(self)))[-3:]}.gif"

        self.filetype = filetype
        self.outfile = pathlib.Path(outdir).resolve() / f"{filename_stem}.{filetype}"
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


def plot_flow_fields(dim_reduced_data, x_direction=0, y_direction=1, grid_n=13, scatter_alpha=0, normalize_method=None, fig=None, axs=None, method='quiver', format_axis=True, limits=None):
    """
    Examples
    --------
    >>> X = np.random.normal(size=(100,2))
    >>> plot_flow_fields({'random points': X}, normalize_method='squares', grid_n=20)
    """
    assert normalize_method in {None, 'none', 'diffs', 'hcubes', 'squares'}
    if fig is None:
        fig, axs = plt.subplots(nrows=1, ncols=len(dim_reduced_data), squeeze=False, layout='tight', figsize=(12,4))
        axs = axs[0]

    for idx, (name, latents) in enumerate(dim_reduced_data.items()):
        e1, e2 = np.zeros(latents.shape[1]), np.zeros(latents.shape[1])
        e1[x_direction] = 1
        e2[y_direction] = 1

        ax: plt.Axes = axs[idx]
        ax.scatter(latents @ e1, latents @ e2, s=5, alpha=scatter_alpha)
        if limits is None:
            x1, x2, y1, y2 = ax.axis()
        else:
            x1, x2, y1, y2 = limits
        x_points = np.linspace(x1, x2, grid_n)
        y_points = np.linspace(y1, y2, grid_n)
        assert x1 < x2 and y1 < y2

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
                    arrow = arrow
                    arrows.append(arrow)
                    origins.append([np.nanmean(x_points[i:i + 2]), np.nanmean(y_points[j:j + 2])])
                    n_points.append(s.sum())
                else:
                    arrow = np.nanmean(d_latents[s],axis=0) * 0
                    arrows.append(arrow)
                    origins.append([np.nanmean(x_points[i:i + 2]), np.nanmean(y_points[j:j + 2])])
                    n_points.append(s.sum())

        origins, arrows, n_points = np.array(origins), np.array(arrows), np.array(n_points)
        arrows = np.array([arrows @ e1, arrows @ e2]).T
        if normalize_method == 'squares':
            arrows = arrows / np.linalg.norm(arrows, axis=1)[:, np.newaxis]

        if method == 'quiver':
            ax.quiver(origins[:, 0], origins[:, 1], arrows[:,0], arrows[:,1], scale=1 / 20, units='dots', color='red')
        elif method == 'streamplot':
            origins = origins.reshape((grid_n-1,grid_n-1,2))
            a = origins[..., 1].mean(axis=0) # -1 2 is the x axis
            b = origins[..., 0].mean(axis=1)
            arrows = arrows.reshape((grid_n-1,grid_n-1,2))
            ax.streamplot(y=a, x=b, v=arrows[...,1].T, u=arrows[...,0].T, color='red')
        else:
            raise ValueError()

        if format_axis:
            ax.axis('scaled')
            ax.axis('off')
        # TODO: this should be a test?
        """
        # I used this for debugging this function:
        # note that the rotation is backwards (clockwise) for this LDS
        from adaptive_latents.input_sources.lds_simulation import LDS
        rng = np.random.default_rng(13)

        lds = LDS.circular_lds(transitions_per_rotation=30, obs_d=2, rng=rng,)
        lds.C = np.eye(2)
        _, X, _ = lds.simulate(1200, initial_state=[0,6], rng=rng)
        X = list(X)
        for _ in range(10):
            X.append(X[-1] - np.array([1,0]))
        X = np.array(X)

        importlib.reload(adaptive_latents.plotting_functions)
        adaptive_latents.plotting_functions.plot_flow_fields(
            {'test':X + np.array([0,0])},
            method='streamplot', normalize_method='hcubes',
            x_direction=0, y_direction=1, scatter_alpha=1,
        )
        """


class MultiRowRunComparison:
    def __init__(self, n_rows, time_in_samples=False, error_plot_multi_color=False, color_sequence='first_special'):
        self.n_rows = n_rows
        self.time_in_samples = time_in_samples
        self.error_plot_multi_color = error_plot_multi_color  # controls if error plots color by component

        self.fig, self.axs = plt.subplots(figsize=(14, 2*n_rows-1), nrows=n_rows, ncols=2, sharex='col', layout='tight', gridspec_kw={'width_ratios': [7, 1]})
        gs = self.axs[0, 1].get_gridspec()
        for a in self.axs[:, 1]:
            a.remove()
        self.axbig = self.fig.add_subplot(gs[:, 1])

        self.entries = [[] for _ in range(n_rows)]

        self.halfway_time = None
        self.common_time_start = None
        self.common_time_end = None
        self.any_time_start = None
        self.any_time_end = None


        self.to_write = [[] for _ in range(n_rows)]
        match color_sequence:
            case 'first_special':
                self.color_sequence = itertools.chain(['C0'], itertools.repeat('k'))
            case 'default_cycle':
                self.color_sequence = (f'C{n}' for n in itertools.cycle(list(range(9))))
            case _:
                raise ValueError()
        self.current_color = next(self.color_sequence)

    def new_set(self):
        self.current_color = next(self.color_sequence)

    def register_entry(self, row_n, plot_type='line', **kwargs):
        self.entries[row_n].append(dict(color=self.current_color, plot_type=plot_type) | kwargs)

    def plot_entries(self):
        for row in self.entries:
            assert len(set([entry.get('ylabel', '') for entry in row])) == 1

        self.any_time_start = min([min(e['to_plot'].t) for row in self.entries for e in row])
        self.any_time_end = max([max(e['to_plot'].t) for row in self.entries for e in row])

        self.common_time_start = max([min(e['to_plot'].t) for row in self.entries for e in row])
        self.common_time_end = min([max(e['to_plot'].t) for row in self.entries for e in row])

        self.halfway_time = (self.common_time_start + self.common_time_end) / 2

        for row_idx in range(self.n_rows):
            for layer_idx, e in enumerate(self.entries[row_idx]):
                ax = self.axs[row_idx, 0]
                match e.pop('plot_type'):
                    case 'line':
                        text, style = self.plot_line_entry(ax=ax, **e)
                    case 'error':
                        text, style = self.plot_error_entry(ax=ax, **e)
                    case _:
                        raise ValueError()


                self.to_write[row_idx].append((layer_idx, text, {'color': e['color']} | style))

        xlabel = 'time' if not self.time_in_samples else 'time (samples)'
        self.axs[-1,0].set_xlabel(xlabel)

        self.set_axlim_and_coord_format()
        self.write_last_half_means(self.to_write)


    def plot_line_entry(self, ax, to_plot, ylabel, color):
        t = to_plot.t
        if self.time_in_samples:
            t = to_plot.t / to_plot.dt
        self.plot_with_trendline(ax, t, to_plot, color)
        ax.set_ylabel(ylabel)

        test_slice = (self.halfway_time < to_plot.t) & (to_plot.t < self.common_time_end)
        last_half_mean = to_plot[test_slice].mean()

        text = f'{last_half_mean:.2f}'
        style = {}

        return text, style

    def plot_error_entry(self, ax, to_plot, true_values, ylabel, color):
        predicted_values = to_plot
        assert (true_values.t == predicted_values.t).all()
        t = true_values.t

        if self.time_in_samples:
            t = t / predicted_values.dt

        for i in range(predicted_values.shape[1]):
            color = f'C{i}' if self.error_plot_multi_color else color
            ax.plot(t, true_values[:, i], color=color)
            ax.plot(t, predicted_values[:, i], color=color, alpha=.5)

        ax.set_ylabel(ylabel)

        test_slice = (self.halfway_time < to_plot.t) & (to_plot.t < self.common_time_end)
        correlations = [np.corrcoef(predicted_values[test_slice, i], true_values[test_slice, i])[0, 1] for i in range(predicted_values.shape[1])]
        text = ' '.join([f'{r:.2f}' for r in correlations] )
        style = {'fontsize': 'x-small'}

        return text, style

    def add_right_text(self, to_write):
        self.axbig.axis("off")
        self.axbig.text(0, 1, to_write, transform=self.axbig.transAxes, verticalalignment="top")

    def write_last_half_means(self, to_write):
        for i, l in enumerate(to_write):
            for idx, text, kw in l:
                x, y = .92, .93 - .1 * idx
                x, y = self.axs[i, 0].transLimits.inverted().transform([x, y])
                self.axs[i, 0].text(x, y, text, clip_on=True, verticalalignment='top', **kw)

    def set_axlim_and_coord_format(self):
        for axis in self.axs[:, 0]:
            data_lim = np.array(axis.dataLim).T.flatten()
            data_lim[0] = self.any_time_start
            data_lim[1] = self.any_time_end
            if np.isfinite(data_lim).all():
                bounds = data_lim
                bounds[:2] = (bounds[:2] - bounds[:2].mean()) * np.array([1.02, 1.2]) + bounds[:2].mean()
                bounds[2:] = (bounds[2:] - bounds[2:].mean()) * np.array([1.05, 1.05]) + bounds[2:].mean()
                axis.axis(bounds)
                axis.format_coord = lambda x, y: 'x={:g}, y={:g}'.format(x, y)

    def write_transformer_comparison(self, transformers):
        to_write = self.transformer_comparison(transformers)
        self.add_right_text(to_write)


    @staticmethod
    def transformer_comparison(transformers, ignore_keys=('input_streams', 'output_streams', 'log_level')):
        types = [type(t) for t in transformers]
        if len(set(types)) == 1:
            params_per_transformer_list = [t.get_params() for t in transformers]
            super_param_dict = {}
            for key in params_per_transformer_list[0].keys():
                values = [p[key] for p in params_per_transformer_list]
                if len(set(values)) == 1:
                    values = values[0]
                    if key in ignore_keys:
                        continue
                super_param_dict[key] = values
            to_write = "\n".join(f"{k}: {v}" for k, v in super_param_dict.items())
        else:
            to_write = "\n".join([str(t.__name__) for t in types])
        return to_write


    @staticmethod
    def _one_sided_ewma(data, com=100):
        import pandas as pd
        # TODO: actually implement this
        return pd.DataFrame(data=dict(data=data)).ewm(com).mean()["data"]

    @classmethod
    def plot_with_trendline(cls, ax, times, data, color, com=100):
        ax.plot(times, data, alpha=.25, color=color)
        smoothed_data = cls._one_sided_ewma(data, com, )
        ax.plot(times, smoothed_data, color=color)


    @staticmethod
    def compare_bw_runs(bws, behavior_dicts=None, t_in_samples=False):
        from adaptive_latents.utils import resample_matched_timeseries

        bws: list[adaptive_latents.Bubblewrap]
        for bw in bws:
            assert bw.log_level >= 2
            assert bw.check_dt

        has_behavior = behavior_dicts is not None
        if not has_behavior:
            behavior_dicts = [{} for _ in range(len(bws))]

        plot = MultiRowRunComparison(n_rows=3+has_behavior, time_in_samples=t_in_samples)

        for bw, behavior_dict  in zip(bws, behavior_dicts):
            to_plot = ArrayWithTime.from_list(bw.log['log_pred_p'])
            plot.register_entry(row_n=0, to_plot=to_plot, ylabel='log_pred_p', plot_type='line')

            to_plot = ArrayWithTime.from_list(bw.log['entropy'])
            plot.register_entry(row_n=1, to_plot=to_plot, ylabel='entropy', plot_type='line')

            to_plot = ArrayWithTime.from_list(bw.log['pred_error'], squeeze_type='to_2d')
            to_plot = (to_plot**2).mean(axis=1)
            plot.register_entry(row_n=2, to_plot=to_plot, ylabel='pred_error (mse)', plot_type='line')

            if has_behavior:
                true_values = resample_matched_timeseries(
                    behavior_dict['true_behavior'],
                    behavior_dict['true_behavior'].t,
                    behavior_dict['predicted_behavior'].t
                )
                predicted_values = behavior_dict['predicted_behavior']

                plot.register_entry(
                    row_n=3,
                    plot_type='error',
                    to_plot = predicted_values,
                    true_values = true_values,
                    ylabel = 'behavior',
                )

            plot.new_set()

        plot.plot_entries()

        max_entropy = np.log2(bw.N)
        plot.axs[1, 0].axhline(max_entropy, color='k', linestyle='--')

        plot.write_transformer_comparison(bws)


    @staticmethod
    def compare_predictor_runs(predictors, t_in_samples=False, color_sequence='default_cycle'):
        predictors: list[adaptive_latents.predictor.Predictor]
        for predictor in predictors:
            assert predictor.log_level >= 2
            assert predictor.check_dt

        plot = MultiRowRunComparison(n_rows=3, time_in_samples=t_in_samples, color_sequence=color_sequence)

        for predictor in predictors:
            to_plot = ArrayWithTime.from_list(predictor.log['log_pred_p'])
            plot.register_entry(row_n=0, to_plot=to_plot, ylabel='log_pred_p', plot_type='line')

            to_plot = ArrayWithTime.from_list(predictor.log['pred_error'], squeeze_type='to_2d')
            to_plot = (to_plot**2).mean(axis=1)
            plot.register_entry(row_n=1, to_plot=to_plot, ylabel='pred_error (mse)', plot_type='line')

            to_plot = ArrayWithTime.from_list(predictor.log['step_time'], squeeze_type='squeeze')
            plot.register_entry(row_n=2, to_plot=to_plot*1000, ylabel='step time (ms)', plot_type='line')

            plot.new_set()

        plot.plot_entries()

        plot.write_transformer_comparison(predictors)
