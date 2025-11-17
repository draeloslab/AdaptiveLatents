import numpy as np

import adaptive_latents
import seaborn as sns
from sim_stim import make_srs, make_slices_tensor
from adaptive_latents import datasets, ArrayWithTime
from adaptive_latents.regressions import BaseKernelRegressor
from matplotlib.path import Path
import matplotlib.pyplot as plt
from adaptive_latents.utils import save_to_cache
import pandas

_vh = .5
verts = [ (-1., -_vh), (-1., _vh), (1., _vh), (1., -_vh), (-1., -_vh), ]
codes = [ Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY, ]
white_bar_path = Path(verts, codes)
violinplot_inner_kws = {'marker': white_bar_path, 'markersize': 3, 'markerfacecolor': 'white', }


def proportion_in_space(desired, designed):
    assert np.allclose(desired.T @ desired, np.eye(desired.shape[1]))
    proj = desired @ desired.T @ designed
    in_norm = np.linalg.norm(proj)
    total_norm = np.linalg.norm(designed)
    if total_norm == 0:
        ratio = 0
    else:
        ratio = in_norm / total_norm
    return ratio

def make_unit(x):
    x = np.squeeze(x)
    assert len(x.shape) == 1
    return x / np.linalg.norm(x)

def angle(a,b):
    return np.acos(make_unit(a) @ make_unit(b).flatten()) * 180/np.pi

def srs_to_l_df(srs):
    records = []
    for k, sr_list in srs.items():
        for sr_i, sr in enumerate(sr_list):
            latents: ArrayWithTime = sr.log['latents']
            for l_i, l in enumerate(sr.stim_designer.log):
                t_of_stim = l['time_of_stim']
                stim_sample = latents.time_to_sample(t_of_stim)
                old_v = latents[stim_sample-1] - latents[stim_sample-2]
                this_v = latents[stim_sample] - latents[stim_sample-1]
                l['old_v'] = old_v
                l['this_v'] = this_v

                records.append(dict(sr_key=k, sr_i=sr_i, l_i=l_i, l=l))
    return pandas.DataFrame(records)


def extract_metrics(srs, preq_cutoff=None, metric_functions=None):
    if metric_functions is None:
        metric_functions = {
            'proportions': lambda l: proportion_in_space(l['v'], l['s']),
            'preq_errors': lambda l: np.linalg.norm(l['observed_s_hat'] - l['stim_reg'].predict(l['observed_reg_input'])) if l['stim_reg'] is not None else np.nan,
            'v_delta_errors': lambda l: proportion_in_space(l['v'], l['observed_s_hat']),
            's_delta_errors': lambda l: np.linalg.norm(l['s'] - l['observed_s_hat']),
            'angles': lambda l: angle(l['observed_s_hat'], l['v']),
            'mags_along': lambda l: l['observed_s_hat'] @ make_unit(l['v']),
            'mags': lambda l: np.linalg.norm(l['observed_s_hat']),
            'alignment_with_old_v': lambda l: angle(l['this_v'], l['old_v']),
            'v_mag_ratio': lambda l: np.linalg.norm(l['this_v']) / np.linalg.norm(l['old_v']),
        }
    metrics = {name: [] for name in metric_functions}

    for k, sr_list in srs.items():
        for m in metrics.values():
            m.append([])

        for sr in sr_list:
            for m in metrics.values():
                m[-1].append([])

            latents: ArrayWithTime = sr.log['latents']

            for l in sr.stim_designer.log:
                t_of_stim = l['time_of_stim']
                stim_sample = latents.time_to_sample(t_of_stim)
                old_v = latents[stim_sample-1] - latents[stim_sample-2]
                this_v = latents[stim_sample] - latents[stim_sample-1]
                l['old_v'] = old_v
                l['this_v'] = this_v

                for name, m in metrics.items():
                    m[-1][-1].append(metric_functions[name](l))


            if preq_cutoff is not None:
                for m in metrics.values():
                    m[-1][-1] = m[-1][-1][:preq_cutoff]

    if preq_cutoff is None:
        preq_cutoff = np.inf
        for a in list(metrics.values())[0]:
            for b in a:
                if len(b) < preq_cutoff:
                    preq_cutoff = len(b)

        for k in metrics:
            metrics[k] = [[b[:preq_cutoff] for b in a] for a in metrics[k]]

    return metrics

def apply_lambda(srs, f, preq_cutoff=None):
    return extract_metrics(srs, preq_cutoff=preq_cutoff, metric_functions={'custom': f})['custom']

def unpack_metrics(metrics):
    if isinstance(metrics, dict):
        return metrics['proportions'], metrics['preq_errors'], metrics['v_delta_errors'], metrics['s_delta_errors'], metrics['angles'], metrics['mags_along'], metrics['mags'], metrics['alignment_with_old_v'], metrics['v_mag_ratio']
    else:
        return metrics


def open_v_closed_plot(srs, proportions, preq_errors, v_delta_errors, s_delta_errors, show_individuals=True, legend=False):
    fig, axs = plt.subplots(ncols=2, nrows=1, squeeze=False, layout='constrained', figsize=(2*4, 1*4))

    ax: plt.Axes = axs[0,0]
    if show_individuals:
        for i, (k, errors) in enumerate(zip(srs.keys(), v_delta_errors)):
            for j, e in enumerate(errors):
                ax.plot(e, color=f'C{i}', alpha=0.1)
    for i, (k, errors) in enumerate(zip(srs.keys(), v_delta_errors)):
        trendline = np.mean(errors, axis=0)
        ax.plot(trendline, color=f'C{i}', lw=1.5, label=f'{k} {trendline[trendline.size//2:].mean():.2f}')
    ax.set_title('$s_{\\text{obs}}$ along $v$')

    ax: plt.Axes = axs[0,1]
    if show_individuals:
        for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
            for j, e in enumerate(errors):
                ax.plot(e, color=f'C{i}', alpha=0.1, )

    for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
        trendline = np.mean(errors, axis=0)
        ax.plot(trendline, color=f'C{i}', lw=1.5, label=f'{k} {trendline[trendline.size//2:].mean():.2f}')
    ax.set_title('$\\Vert \\hat s_{obs} - \\hat S_{i-1}(x_i, u_i, t_i) \\Vert$')

    if legend:
        for ax in axs.flatten():
            ax.legend()


    return fig

N = 5

def plot_optim_col_vs_rand_with_high_d_rand():
    @save_to_cache('optim_col_vs_rand_with_high_d_rand')
    def to_cache(n_runs=N):
        d = datasets.Odoherty21Dataset()
        data = d.neural_data
        srs = make_srs(data=data, rng=rng, comparison_preset='optim_col_vs_rand_with_high_d_rand', n_runs=n_runs, show_tqdm=True)
        return srs

    srs = to_cache(n_runs=N, _recalculate_cache_value=False)

    l_df = srs_to_l_df(srs)
    l_df[['optim_method', 'stim_direction_type']] = l_df['sr_key'].str.split(' ', expand=True)

    for stim_direction_type_to_drop in [
        'random+',
        'col'
    ]:
        l_df.drop(index=l_df.index[(l_df.stim_direction_type == stim_direction_type_to_drop)], inplace=True)


    order = ('-ones', 'ones', 'random', 'random_feasible', 'first')
    l_df.sort_values(by='stim_direction_type', inplace=True, key=lambda x: x.apply(order.index))

    stim_direction_types = l_df.stim_direction_type.unique()

    ncols = 3
    fig, axs = plt.subplots(ncols=ncols, nrows=len(stim_direction_types), squeeze=False, figsize=(4*ncols, 4*len(stim_direction_types)), layout='constrained', sharey='col')

    fig5, ax5 = plt.subplots(figsize=(8, 8))


    make_whole_plots = False
    for row, stim_direction_type in enumerate(stim_direction_types):
        sub_df = pandas.DataFrame(l_df[l_df['stim_direction_type'] == stim_direction_type])

        ax: plt.Axes = axs[row, 0]
        sub_df['angles(s_obs,v)'] = sub_df.l.apply(lambda l: angle(l['observed_s_hat'], l['v']))
        if make_whole_plots:
            sns.violinplot(sub_df, x='optim_method', y='angles(s_obs,v)', orient='v', ax=ax, width=1, density_norm='width',inner_kws = violinplot_inner_kws)
            sns.swarmplot(sub_df, x='optim_method', y='angles(s_obs,v)', orient='v', ax=ax, size=1, edgecolor='white')
        ax.set_title(f's_obs angle from v={{{stim_direction_type}}}')
        ax.set_ylabel('cosine angle (degrees)')

        if stim_direction_type == 'first':
            order = ('single', 'many', 'shuffled', 'normal')
            sub_df.sort_values(by='optim_method', inplace=True, key=lambda x: x.apply(order.index))

            fig4, ax4 = plt.subplots(figsize=(8, 8))
            sns.violinplot(sub_df, x='optim_method', y='angles(s_obs,v)', orient='v', ax=ax4, width=1, density_norm='width',inner_kws = violinplot_inner_kws)
            # sns.swarmplot(sub_df, x='optim_method', y='angles(s_obs,v)', orient='v', ax=ax4, size=1, edgecolor='white')
            ax4.set_title(f's_obs angle from v={{{stim_direction_type}}}')


        ax: plt.Axes = axs[row, 1]
        metric_name = 'angles(s_designed,v)'
        sub_df[metric_name] = sub_df.l.apply(lambda l: angle(l['s'], l['v']))
        just_normal_sub_df = sub_df[(sub_df['optim_method'] == 'normal')]
        if make_whole_plots:
            sns.violinplot(just_normal_sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax, width=1, density_norm='width',inner_kws = violinplot_inner_kws)
            sns.swarmplot(just_normal_sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax, size=1, edgecolor='white')
        ax.set_title(f's_designed angle with v={{{stim_direction_type}}}')

        ax: plt.Axes = axs[row, 2]
        if make_whole_plots:
            sns.scatterplot(just_normal_sub_df, x='angles(s_designed,v)', y='angles(s_obs,v)', ax=ax)
        ax.plot([0,120], [0,120], 'k')
        ax.set_xlim([0, 120])
        ax.set_ylim([0, 120])
        ax.axis('equal')

        sns.scatterplot(just_normal_sub_df, x='angles(s_designed,v)', y='angles(s_obs,v)', zorder=10-row, ax=ax5, label=stim_direction_type)
        print((just_normal_sub_df['angles(s_designed,v)'] > just_normal_sub_df['angles(s_obs,v)']).sum())
        print((just_normal_sub_df['angles(s_designed,v)'] == just_normal_sub_df['angles(s_obs,v)']).sum())
        print((just_normal_sub_df['angles(s_designed,v)'] < just_normal_sub_df['angles(s_obs,v)']).sum())
        breakpoint()
        ax5.plot([0,120], [0,120], 'k')
        ax5.set_xlim([0, 120])
        ax5.set_ylim([0, 120])
    ax5.legend()

    order = ('first', '-ones', 'ones', 'random', 'random_feasible')
    l_df.sort_values(by='stim_direction_type', inplace=True, key=lambda x: x.apply(order.index))

    fig2, ax2 = plt.subplots(ncols=2, nrows=4, figsize=(6 * 2, 4 * 4), squeeze=False, layout='constrained')

    for row, optim_method in enumerate(('normal', 'shuffled', 'many', 'single')):
        sub_df = l_df[(l_df['optim_method'] == optim_method)]

        if optim_method == 'normal':
            ax: plt.Axes = ax2[row, 0]
            metric_name = 'angles(s_designed,v)'
            sub_df[metric_name] = sub_df.l.apply(lambda l: angle(l['s'], l['v']))
            if make_whole_plots:
                sns.violinplot(sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax, width=1, density_norm='width',inner_kws = violinplot_inner_kws)
                sns.swarmplot(sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax, size=1, edgecolor='white')
            ax.set_title(f'{optim_method=}')

            fig3, ax3 = plt.subplots(figsize=(8, 8))
            sns.violinplot(sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax3, scale='width', width=1, density_norm='width',inner_kws = violinplot_inner_kws)
            # sns.swarmplot(sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax3, size=2, edgecolor='white', color='C0')


        ax: plt.Axes = ax2[row, 1]
        metric_name = 'angles(s_obs,v)'
        sub_df[metric_name] = sub_df.l.apply(lambda l: angle(l['observed_s_hat'], l['v']))
        if make_whole_plots:
            sns.violinplot(sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax, width=1, density_norm='width',inner_kws = violinplot_inner_kws)
            sns.swarmplot(sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax, size=1, edgecolor='white')
        ax.set_title(f'{optim_method=}')



    return fig, [fig2, fig3, fig4, fig5]



def plot_optim_col_vs_rand_with_high_d_rand_closed():
    @save_to_cache('optim_col_vs_rand_with_high_d_rand_closed')
    def to_cache(n_runs=N):
        d = datasets.Odoherty21Dataset()
        data = d.neural_data
        srs = make_srs(data=data, rng=rng, comparison_preset='optim_col_vs_rand_with_high_d_rand_closed', n_runs=n_runs, show_tqdm=True)
        return srs

    srs = to_cache(n_runs=1, _recalculate_cache_value=False)

    l_df = srs_to_l_df(srs)
    l_df[['optim_method', 'stim_direction_type']] = l_df['sr_key'].str.split(' ', expand=True)

    l_df.drop(index=l_df.index[(l_df.stim_direction_type == 'random+')], inplace=True)
    l_df.drop(index=l_df.index[(l_df.stim_direction_type == 'col')], inplace=True)
    l_df.drop(index=l_df.index[(l_df.l_i <= 10)], inplace=True)
    stim_direction_types = l_df.stim_direction_type.unique()

    ncols = 3
    fig, axs = plt.subplots(ncols=ncols, nrows=len(stim_direction_types), squeeze=False, figsize=(4*ncols, 4*len(stim_direction_types)), layout='constrained', sharey='col')

    fig5, ax5 = plt.subplots(figsize=(8, 8))


    make_whole_plots = False
    for row, stim_direction_type in enumerate(stim_direction_types):
        sub_df = pandas.DataFrame(l_df[l_df['stim_direction_type'] == stim_direction_type])

        ax: plt.Axes = axs[row, 0]
        sub_df['angles(s_obs,v)'] = sub_df.l.apply(lambda l: angle(l['observed_s_hat'], l['v']))
        if make_whole_plots:
            sns.violinplot(sub_df, x='optim_method', y='angles(s_obs,v)', orient='v', ax=ax, width=1, density_norm='width',inner_kws = violinplot_inner_kws)
            sns.swarmplot(sub_df, x='optim_method', y='angles(s_obs,v)', orient='v', ax=ax, size=1, edgecolor='white')
        ax.set_title(f's_obs angle from v={{{stim_direction_type}}}')
        ax.set_ylabel('cosine angle (degrees)')

        if row == 0:
            fig4, ax4 = plt.subplots(figsize=(8, 8))
            sns.violinplot(sub_df, x='optim_method', y='angles(s_obs,v)', orient='v', ax=ax4, width=1, density_norm='width',inner_kws = violinplot_inner_kws)
            # sns.swarmplot(sub_df, x='optim_method', y='angles(s_obs,v)', orient='v', ax=ax4, size=1, edgecolor='white')
            ax4.set_title(f's_obs angle from v={{{stim_direction_type}}}')


        ax: plt.Axes = axs[row, 1]
        metric_name = 'angles(s_designed,v)'
        sub_df[metric_name] = sub_df.l.apply(lambda l: angle(l['s'], l['v']))
        just_normal_sub_df = sub_df[(sub_df['optim_method'] == 'normal')]
        if make_whole_plots:
            sns.violinplot(just_normal_sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax, width=1, density_norm='width',inner_kws = violinplot_inner_kws)
            sns.swarmplot(just_normal_sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax, size=1, edgecolor='white')
        ax.set_title(f's_designed angle with v={{{stim_direction_type}}}')

        ax: plt.Axes = axs[row, 2]
        if make_whole_plots:
            sns.scatterplot(just_normal_sub_df, x='angles(s_designed,v)', y='angles(s_obs,v)', ax=ax)
        ax.plot([0,120], [0,120], 'k')
        ax.set_xlim([0, 120])
        ax.set_ylim([0, 120])
        ax.axis('equal')

        sns.scatterplot(just_normal_sub_df, x='angles(s_designed,v)', y='angles(s_obs,v)', zorder=10-row, ax=ax5, label=stim_direction_type)
        ax5.plot([0,120], [0,120], 'k')
        ax5.set_xlim([0, 120])
        ax5.set_ylim([0, 120])
    ax5.legend()



    fig2, ax2 = plt.subplots(ncols=2, nrows=4, figsize=(6 * 2, 4 * 4), squeeze=False, layout='constrained')

    for row, optim_method in enumerate(('normal', 'shuffled', 'many', 'single')):
        sub_df = l_df[(l_df['optim_method'] == optim_method)]

        if optim_method == 'normal':
            ax: plt.Axes = ax2[row, 0]
            metric_name = 'angles(s_designed,v)'
            sub_df[metric_name] = sub_df.l.apply(lambda l: angle(l['s'], l['v']))
            if make_whole_plots:
                sns.violinplot(sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax, width=1, density_norm='width',inner_kws = violinplot_inner_kws)
                sns.swarmplot(sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax, size=1, edgecolor='white')
            ax.set_title(f'{optim_method=}')

            fig3, ax3 = plt.subplots(figsize=(8, 8))
            sns.violinplot(sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax3, scale='width', width=1, density_norm='width',inner_kws = violinplot_inner_kws)
            # sns.swarmplot(sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax3, size=2, edgecolor='white', color='C0')


        ax: plt.Axes = ax2[row, 1]
        metric_name = 'angles(s_obs,v)'
        sub_df[metric_name] = sub_df.l.apply(lambda l: angle(l['observed_s_hat'], l['v']))
        if make_whole_plots:
            sns.violinplot(sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax, width=1, density_norm='width',inner_kws = violinplot_inner_kws)
            sns.swarmplot(sub_df, x='stim_direction_type', y=metric_name, orient='v', ax=ax, size=1, edgecolor='white')
        ax.set_title(f'{optim_method=}')



    return fig, [fig2, fig3, fig4, fig5]

def plot_optim_open_vs_closed(args):
    @save_to_cache('optim_open_vs_closed')
    def f(n_runs=N, args_dataset=args.dataset, args_type_of_dim_red=args.type_of_dim_red, args_type_of_autoreg=args.type_of_autoreg):
        if args_dataset == 'odoherty21':
            data = datasets.Odoherty21Dataset().neural_data
        elif args_dataset == 'zong22':
            data = datasets.Zong22Dataset().neural_data
        else:
            raise ValueError()
        
        if args_type_of_autoreg == 'kf':
            from adaptive_latents import StreamingKalmanFilter
            autoreg = StreamingKalmanFilter
        elif args_type_of_autoreg == 'bw':
            from adaptive_latents import Bubblewrap
            autoreg = Bubblewrap
        elif args_type_of_autoreg == 'vjf':
            from adaptive_latents import VJF
            autoreg = VJF
        
        srs = make_srs(data=data, rng=rng, comparison_preset='optim_open_vs_closed', n_runs=n_runs, show_tqdm=True, overrides=dict(last_dim_red=args_type_of_dim_red, autoreg=autoreg))
        return srs

    srs = f()

    proportions, preq_errors, v_delta_errors, s_delta_errors, angles, mags_along, mags, alignment_with_old_v, v_mag_ratio = unpack_metrics(
        extract_metrics(srs, preq_cutoff=None))
    fig = open_v_closed_plot(srs, proportions, preq_errors, v_delta_errors, s_delta_errors, show_individuals=False, legend=True)
    fig.axes[0].set_title(f'$s_{{\\text{{obs}}}}$ along $v$, {args.dataset} {args.type_of_dim_red} {args.type_of_autoreg}')
 
    l_df = srs_to_l_df(srs)
    fig2, axs = plt.subplots(ncols=2, squeeze=False, figsize=(10,4), layout='constrained')
    l_df[['open_closed', 'true_s']] = l_df['sr_key'].str.split(' ', expand=True)

    l_df['angle(s_obs,v)'] = l_df.l.apply(lambda l: angle(l['observed_s_hat'], l['v']))
    l_df['s_obs along v'] = l_df.l.apply(lambda l: proportion_in_space(l['v'], l['observed_s_hat']))


    sns.violinplot(data=l_df[l_df['l_i'] > 20], x='sr_key', y='angle(s_obs,v)', ax=axs[0,0], density_norm='width',inner_kws = violinplot_inner_kws) #  density_norm='width',
    sns.violinplot(data=l_df[l_df['l_i'] > 20], x='sr_key', y='s_obs along v', ax=axs[0,1], density_norm='width',inner_kws = violinplot_inner_kws)



    return fig, [fig2]

def plot_optim_open_vs_closed_toy():
    @save_to_cache('optim_open_vs_closed_toy')
    def f():
        n_revolutions = 80
        obs_d = 130

        rng = np.random.default_rng(4)
        from adaptive_latents.input_sources.lds_simulation import LDS

        all_srs = []
        for _ in range(N):
            lds = LDS.circular_lds(rng=rng, obs_d=obs_d)
            _, data, _ = lds.simulate(int(lds.transitions_per_rotation * n_revolutions), rng=rng, initial_state=np.array([20, 0]))
            t = np.arange(data.shape[0]) * 1 / lds.transitions_per_rotation
            data = ArrayWithTime(data, t)

            srs = make_srs(data=data, rng=rng, comparison_preset='optim_open_vs_closed_toy', n_runs=1, show_tqdm=True, overrides=dict(last_dim_red=args.type_of_dim_red))
            all_srs.append(srs)

        srs = {k: [sub_srs[k][0] for sub_srs in all_srs] for k in srs.keys()}
        return srs

    srs = f()

    proportions, preq_errors, v_delta_errors, s_delta_errors, angles, mags_along, mags, alignment_with_old_v, v_mag_ratio = unpack_metrics(
        extract_metrics(srs, preq_cutoff=None))
    fig = open_v_closed_plot(srs, proportions, preq_errors, v_delta_errors, s_delta_errors, show_individuals=False)


    l_df = srs_to_l_df(srs)
    fig2, axs = plt.subplots(ncols=2, squeeze=False, layout='constrained')
    l_df[['open_closed', 'true_s']] = l_df['sr_key'].str.split(' ', expand=True)

    l_df['angle(s_obs,v)'] = l_df.l.apply(lambda l: angle(l['observed_s_hat'], l['v']))
    l_df['s_obs along v'] = l_df.l.apply(lambda l: proportion_in_space(l['v'], l['observed_s_hat']))
    sns.violinplot(data=l_df[l_df['l_i'] > 20], x='sr_key', y='angle(s_obs,v)', ax=axs[0,0], width=1, density_norm='width',inner_kws = violinplot_inner_kws)
    sns.violinplot(data=l_df[l_df['l_i'] > 20], x='sr_key', y='s_obs along v', ax=axs[0,1], width=1, density_norm='width',inner_kws = violinplot_inner_kws)

    return fig, [fig2]

if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    parser.add_argument( "--type-of-plot", type=str, required=True)
    parser.add_argument( "--type-of-dim-red", type=str, required=False)
    parser.add_argument( "--type-of-autoreg", type=str, required=False, default='kf')
    parser.add_argument( "--dataset", type=str, required=False, default='Odoherty21')
    args = parser.parse_args()

    rng = np.random.default_rng(0)

    match args.type_of_plot:
        case 'optim_col_vs_rand':
            d = datasets.Odoherty21Dataset()
            data = d.neural_data
            srs = make_srs(data=data, rng=rng, comparison_preset='optim_col_vs_rand', n_runs=N, show_tqdm=True)


            proportions, preq_errors, v_delta_errors, s_delta_errors, angles, mags_along, mags, alignment_with_old_v, v_mag_ratio = unpack_metrics(extract_metrics(srs, preq_cutoff=50))

            fig, axs = plt.subplots(ncols=2, squeeze=False, figsize=(8,4), layout='constrained')
            to_plot = {k:v for k, v in zip(srs.keys(), [x[0] for x in proportions])}
            sns.violinplot(to_plot, orient='v', ax=axs[0,0], width=1, density_norm='width',inner_kws = violinplot_inner_kws)
            sns.swarmplot(to_plot, orient='v', ax=axs[0,0])

            for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
                for j, e in enumerate(errors):
                    axs[0,1].plot(e, color=f'C{i}', alpha=0.1)
            for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
                trendline = np.mean(errors, axis=0)
                axs[0,1].plot(trendline, color=f'C{i}', lw=1.5)
            axs[0, 1].semilogy()


        case 'optim_col_vs_rand_with_high_d_rand':
            fig, extra_figs = plot_optim_col_vs_rand_with_high_d_rand()
            for i, extra_fig in enumerate(extra_figs):
                extra_fig.savefig(args.output.with_stem(args.output.stem +f'_extra_{i}'), bbox_inches="tight")

        case 'optim_col_vs_rand_with_high_d_rand_closed':
            fig, extra_figs = plot_optim_col_vs_rand_with_high_d_rand_closed()
            for i, extra_fig in enumerate(extra_figs):
                extra_fig.savefig(args.output.with_stem(args.output.stem +f'_extra_{i}'), bbox_inches="tight")

        case 'optim_open_vs_closed':
            fig, extra_figs = plot_optim_open_vs_closed(args)

            for i, extra_fig in enumerate(extra_figs):
                extra_fig.savefig(args.output.with_stem(args.output.stem + f'_extra_{i}'), bbox_inches="tight")

        case 'optim_open_vs_closed_toy':
            fig, extra_figs = plot_optim_open_vs_closed_toy()
            for i, extra_fig in enumerate(extra_figs):
                extra_fig.savefig(args.output.with_stem(args.output.stem + f'_extra_{i}'), bbox_inches="tight")
        case _:
            raise ValueError()


    fig.savefig(args.output, bbox_inches="tight")
