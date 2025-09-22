import numpy as np

import adaptive_latents
import seaborn as sns
from sim_stim import make_srs, make_slices_tensor
from adaptive_latents import datasets, ArrayWithTime
from adaptive_latents.regressions import BaseKernelRegressor
import matplotlib.pyplot as plt
from adaptive_latents.utils import save_to_cache
import pandas


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


def open_v_closed_plot(srs, proportions, preq_errors, v_delta_errors, s_delta_errors, show_individuals=True):
    fig, axs = plt.subplots(ncols=2, nrows=1, squeeze=False, layout='constrained', figsize=(2*4, 1*4))

    # ax: plt.Axes = axs[0,0]
    # for i, (k, errors) in enumerate(zip(srs.keys(), proportions)):
    #     for j, e in enumerate(errors):
    #         ax.plot(e, color=f'C{i}', alpha=0.1)
    # for i, (k, errors) in enumerate(zip(srs.keys(), proportions)):
    #     trendline = np.mean(errors, axis=0)
    #     ax.plot(trendline, color=f'C{i}', lw=1.5)
    # ax.set_title('$s$ along $v$')


    ax: plt.Axes = axs[0,0]
    if show_individuals:
        for i, (k, errors) in enumerate(zip(srs.keys(), v_delta_errors)):
            for j, e in enumerate(errors):
                ax.plot(e, color=f'C{i}', alpha=0.1)
    for i, (k, errors) in enumerate(zip(srs.keys(), v_delta_errors)):
        trendline = np.mean(errors, axis=0)
        ax.plot(trendline, color=f'C{i}', lw=1.5)
    ax.set_title('$\\hat s_n$ along $v$')

    # ax: plt.Axes = axs[1,1]
    # for i, (k, errors) in enumerate(zip(srs.keys(), s_delta_errors)):
    #     for j, e in enumerate(errors):
    #         ax.plot(e, color=f'C{i}', alpha=0.1)
    # for i, (k, errors) in enumerate(zip(srs.keys(), s_delta_errors)):
    #     trendline = np.mean(errors, axis=0)
    #     ax.plot(trendline, color=f'C{i}', lw=1.5)
    # ax.set_title('$\\Vert s - \\hat s_n \\Vert$')


    ax: plt.Axes = axs[0,1]
    if show_individuals:
        for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
            for j, e in enumerate(errors):
                ax.plot(e, color=f'C{i}', alpha=0.1)

    for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
        trendline = np.mean(errors, axis=0)
        ax.plot(trendline, color=f'C{i}', lw=1.5)
    ax.set_title('$\\Vert \\hat s_n - \\hat S_{n-1}(x_n, u_n) \\Vert$')
    # ax.semilogy()

    return fig
N = 10

def plot_optim_col_vs_rand_with_high_d_rand():
    @save_to_cache('optim_col_vs_rand_with_high_d_rand')
    def to_cache(n_runs=2):
        d = datasets.Odoherty21Dataset()
        data = d.neural_data
        srs = make_srs(data=data, rng=rng, comparison_preset='optim_col_vs_rand_with_high_d_rand', n_runs=n_runs, show_tqdm=True)
        return srs

    srs = to_cache(n_runs=1, _recalculate_cache_value=False)
    stim_direction_types = ('first', 'col', 'random', 'ones', '-ones', 'random+')
    ncols = 6
    fig, axs = plt.subplots(ncols=ncols, nrows=len(stim_direction_types), squeeze=False, figsize=(4*ncols, 4*len(stim_direction_types)), layout='constrained', sharey='col')

    l_df = srs_to_l_df(srs)
    l_df[['optim_method', 'stim_direction_type']] = l_df['sr_key'].str.split(' ', expand=True)

    for row, stim_direction_type in enumerate(stim_direction_types):
        sub_df = pandas.DataFrame(l_df[l_df['stim_direction_type'] == stim_direction_type])
        # sub_srs = {k.split(' ')[0] :v for k, v in srs.items() if stim_direction_type in k}
        # metrics = extract_metrics(sub_srs)

        # sub_srs['normal'] = sub_srs.pop('normal')
        # sub_srs['normal, shuf'] = sub_srs.pop('shuffled')
        # sub_srs['rand 30'] = sub_srs.pop('many')
        # sub_srs['rand 1'] = sub_srs.pop('single')

        ax: plt.Axes = axs[row, 0]
        sub_df['angles(s_obs,v)'] = sub_df.l.apply(lambda l: angle(l['observed_s_hat'], l['v']))
        sns.violinplot(sub_df, x='optim_method', y='angles(s_obs,v)', orient='v', ax=ax)
        sns.swarmplot(sub_df, x='optim_method', y='angles(s_obs,v)', orient='v', ax=ax, size=1, edgecolor='white')
        ax.set_title(f's_obs angle from v={{{stim_direction_type}}}')
        ax.set_ylabel('cosine angle (degrees)')

        # ax: plt.Axes = axs[row, 2]
        # sub_df['angles(s_obs,v)'] = sub_df.l.apply(lambda l: proportion_in_space(l['v'], l['observed_s_hat']))
        # to_plot = {k: np.array(v)[:,10:].flatten() for k, v in zip(sub_srs.keys(), metrics['v_delta_errors'])}
        # sns.violinplot(to_plot, orient='v', ax=ax)
        # sns.swarmplot(to_plot, orient='v', ax=ax, size=1, edgecolor='white')
        # ax.set_title(f's_obs prop. in v={{{stim_direction_type}}} (4b) (10:)')

        # ax: plt.Axes = axs[row, 3]
        # to_plot = {k: np.array(v).flatten() for k, v in zip(sub_srs.keys(), metrics['mags'])}
        # sns.violinplot(to_plot, orient='v', ax=ax)
        # sns.swarmplot(to_plot, orient='v', ax=ax, size=1, edgecolor='white')
        # ax.set_title('s_obs total magnitude')
        # ax.set_ylabel('magnitude (a.u.)')

        ax: plt.Axes = axs[row, 4]
        metric_name = 'angles(s_designed,v)'
        sub_df[metric_name] = sub_df.l.apply(lambda l: angle(l['s'], l['v']))
        just_normal_sub_df = sub_df[(sub_df['optim_method'] == 'normal')]
        sns.violinplot(just_normal_sub_df, x='sr_key', y=metric_name, orient='v', ax=ax)
        sns.swarmplot(just_normal_sub_df, x='sr_key', y=metric_name, orient='v', ax=ax, size=1, edgecolor='white')
        ax.set_title(f's_designed angle with v={{{stim_direction_type}}}')

        ax: plt.Axes = axs[row, 5]
        sns.scatterplot(just_normal_sub_df, x='angles(s_obs,v)', y='angles(s_designed,v)', ax=ax)
        ax.axis('equal')


        # ax: plt.Axes = axs[row, 4]
        # to_plot = {k: np.array(v).flatten() for k, v in zip(sub_srs.keys(), proportions_new)}
        # sns.violinplot(to_plot, orient='v', ax=ax)
        # sns.swarmplot(to_plot, orient='v', ax=ax, size=1, edgecolor='white')
        # ax.set_title('metric 1 from paper')

    return fig, fig


if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    parser.add_argument( "--type-of-plot", type=str, required=True)
    parser.add_argument( "--type-of-dim-red", type=str, required=False)
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
            sns.violinplot(to_plot, orient='v', ax=axs[0,0])
            sns.swarmplot(to_plot, orient='v', ax=axs[0,0])

            for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
                for j, e in enumerate(errors):
                    axs[0,1].plot(e, color=f'C{i}', alpha=0.1)
            for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
                trendline = np.mean(errors, axis=0)
                axs[0,1].plot(trendline, color=f'C{i}', lw=1.5)
            axs[0, 1].semilogy()

        case 'optim_col_vs_rand_with_high_d_rand':
            fig, fig2 = plot_optim_col_vs_rand_with_high_d_rand()
            fig2.savefig(args.output.with_stem('optim_col_vs_rand_with_high_d_rand_T'), bbox_inches="tight")

        case 'optim_open_vs_closed':
            data = datasets.Odoherty21Dataset().neural_data
            srs = make_srs(data=data, rng=rng, comparison_preset='optim_open_vs_closed', n_runs=N, show_tqdm=True, overrides=dict(last_dim_red=args.type_of_dim_red))

            proportions, preq_errors, v_delta_errors, s_delta_errors, angles, mags_along, mags, alignment_with_old_v, v_mag_ratio = unpack_metrics(extract_metrics(srs, preq_cutoff=None))
            fig = open_v_closed_plot(srs, proportions, preq_errors, v_delta_errors, s_delta_errors, show_individuals=False)

        case 'optim_open_vs_closed_toy':
            n_revolutions = 80
            obs_d = 130

            rng = np.random.default_rng(4)
            from adaptive_latents.input_sources.lds_simulation import LDS

            all_srs = []
            for _ in range(N):
                lds = LDS.circular_lds(rng=rng, obs_d=obs_d)
                _, data, _ = lds.simulate(int(lds.transitions_per_rotation * n_revolutions), rng=rng, initial_state=np.array([20, 0]))
                t = np.arange(data.shape[0]) * 1/lds.transitions_per_rotation
                data = ArrayWithTime(data,t)

                srs = make_srs(data=data, rng=rng, comparison_preset='optim_open_vs_closed_toy', n_runs=1, show_tqdm=True, overrides=dict(last_dim_red=args.type_of_dim_red))
                all_srs.append(srs)
            srs = {k: [sub_srs[k][0] for sub_srs in all_srs] for k in srs.keys()}
            proportions, preq_errors, v_delta_errors, s_delta_errors, angles, mags_along, mags, alignment_with_old_v, v_mag_ratio = unpack_metrics(extract_metrics(srs, preq_cutoff=None))
            fig = open_v_closed_plot(srs, proportions, preq_errors, v_delta_errors, s_delta_errors, show_individuals=False)
        case _:
            raise ValueError()


    fig.savefig(args.output, bbox_inches="tight")
