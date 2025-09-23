import numpy as np
from adaptive_latents import ArrayWithTime, StimRegressor
import pandas as pd

import matplotlib.pyplot as plt


def plot_onestep_pred_error_decreasing(srs, row_info, make_slices_tensor):
    fig, axs = plt.subplots(nrows=len(row_info), layout='tight', figsize=(8, 2*len(row_info)+1), sharex=True, sharey=True)

    def p(ax, time_slice_type, space_slice_type, xlabel='time', sr_kind_keys=None, title=None, time_slice=None, last_half_average=False):
        if time_slice is  None:
            time_slice = slice(None, None)

        if sr_kind_keys is None:
            sr_kind_keys = srs.keys()
        for idx, sr_kind_key in reversed(list(enumerate(sr_kind_keys))):
            all_to_plot = []
            for sr in srs[sr_kind_key]:
                run_to_plot = make_slices_tensor(sr)
                sub_to_plot = run_to_plot[time_slice_type][space_slice_type].slice_by_time(time_slice)
                sub_to_plot = ArrayWithTime(np.linalg.norm(sub_to_plot, axis=1), sub_to_plot.t)
                all_to_plot.append(sub_to_plot)
            to_plot = ArrayWithTime(np.hstack(all_to_plot), np.hstack([p.t for p in all_to_plot])) # TODO: sort by time
            # TODO: you could do smoothing here
            ax.plot(to_plot.t, to_plot, '.-', color=f'C{idx}', label=sr_kind_key)
            if last_half_average:
                halfway = (to_plot.t.max() + to_plot.t.min()) / 2
                mean = float(to_plot.slice_by_time(slice(halfway, None)).mean())
                ax.axhline(mean, linestyle='--', color=f'C{idx}')
        # ax.legend(loc='upper right')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('error norm')
        if title is None:
            title = f"time:'{time_slice_type}' space:'{space_slice_type}' norm error"
        ax.set_title(title)

    for idx, values in enumerate(row_info):
        p(ax=axs[idx], **values)

    return fig


def make_table(srs, time_slices, space_slices, make_slices_tensor, save_table=False, normalize_key=None, show_rows=False):
    rows = []

    time_ranges = []
    for sr in [leaf for tree in srs.values() for leaf in tree]:
        t = sr.log['pred_error'].t
        time_ranges.append((t.min(), t.max()))
    time_ranges = np.array(time_ranges)
    time_slice = slice(time_ranges[:,0].min(), time_ranges[:,1].max())

    rmse_func = lambda x: np.sqrt(np.nanmean((x.slice_by_time(time_slice) ** 2)))

    def add_row(sr, title):
        error = make_slices_tensor(sr)
        rmses = error.apply(rmse_func)
        for_row = []
        local_for_row_keys = []
        for ts in time_slices:
            for sp_s in space_slices:
                for_row.append(rmses[ts][sp_s])
                local_for_row_keys.append((ts, sp_s))
        row = pd.DataFrame([for_row], columns=pd.Index(local_for_row_keys))
        row.insert(0, 'run kind', title)
        rows.append(row)

    for title, sr_list in srs.items():
        for sr in sr_list:
            add_row(sr, title)

    df = pd.concat(rows)
    df.reset_index(inplace=True, drop=True)

    if save_table:
        df.to_pickle('generated/debug/table.pkl')

    means = df.groupby('run kind').mean()

    if show_rows:
        table = df.to_latex(index=False, multicolumn_format='c', column_format='c'*len(df.columns))
    else:
        if normalize_key is not None:
            normalized_means = means / means.loc[normalize_key]
            df_to_make_table = normalized_means
        else:
            df_to_make_table = means
        table = df_to_make_table.to_latex(index=True, multicolumn_format='c', column_format='c'*(len(df.columns)+1), float_format="{:0.2f}".format)

    return table, df, means


def plot_manifold_error(srs, comparison_keys=('learning from stim', 'unaware of stim')):
    fig, ax = plt.subplots()

    for k in comparison_keys:
        for sr in srs[k]:
            she = ArrayWithTime.from_list(sr.log['s_hat_error'])
            stim_samples = sr.log['stim_intended_samples']
            she, _ = ArrayWithTime.align_indices(she, stim_samples)
            ax.plot(she, label=k)

    ax.set_xlim([0, 50])
    ax.set_xlabel('# of stimuli')
    ax.set_ylabel(r'~$\mathbb{E}\Vert \hat S - S \Vert$')
    ax.set_ylim(bottom=0)
    ax.legend()

    return fig