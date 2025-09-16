import numpy as np

import adaptive_latents
import seaborn as sns
from sim_stim import make_srs, make_slices_tensor
from adaptive_latents import datasets, ArrayWithTime
from adaptive_latents.regressions import BaseKernelRegressor
import matplotlib.pyplot as plt


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


def extract_metrics_depreciated(srs, preq_cutoff=50):
    proportions = []
    preq_errors = []
    for k, sr_list in srs.items():
        preq_errors.append([])
        proportions.append([])

        for sr in sr_list:
            preq_errors[-1].append([])
            proportions[-1].append([])

            old_stim_reg = None
            for l in sr.stim_designer.log:
                s = l['s']
                v = l['v']
                proportion = proportion_in_space(v, s)
                proportions[-1][-1].append(proportion)

                stim_reg = l['stim_reg']

                if old_stim_reg is not None:
                    # most_recent_row = stim_reg.history[stim_reg.n_observed - 1, :]
                    # i = most_recent_row[:stim_reg.input_d]
                    # o = most_recent_row[stim_reg.input_d:]
                    #
                    # if old_stim_reg.history is None:
                    preq_error = np.nan
                    # else:
                    #     preq_error = np.linalg.norm(o - old_stim_reg.predict(i))
                    preq_errors[-1][-1].append(preq_error)

                old_stim_reg = stim_reg
            assert len(preq_errors[-1][ -1]) >= preq_cutoff, f"to make the array non-ragged, we need to have at least {preq_cutoff} preq errors (not {len(preq_errors[-1][-1])})"
            preq_errors[-1][-1] = np.array(preq_errors[-1][-1][:preq_cutoff])
    return proportions, preq_errors

def extract_metrics(srs, preq_cutoff=None):
    proportions = []
    preq_errors = []
    v_delta_errors = []
    s_delta_errors = []
    angles = []
    mags_along = []
    mags = []
    alignment_with_old_v = []
    v_mag_ratio = []
    for k, sr_list in srs.items():
        preq_errors.append([])
        v_delta_errors.append([])
        s_delta_errors.append([])
        angles.append([])
        mags_along.append([])
        mags.append([])
        alignment_with_old_v.append([])
        v_mag_ratio.append([])
        proportions.append([])
        for sr in sr_list:
            preq_errors[-1].append([])
            v_delta_errors[-1].append([])
            s_delta_errors[-1].append([])
            angles[-1].append([])
            mags_along[-1].append([])
            mags[-1].append([])
            alignment_with_old_v[-1].append([])
            v_mag_ratio[-1].append([])
            proportions[-1].append([])

            high_d_without_stim = sr.log['high_d_without_stim']
            high_d_stims = sr.log['high_d_stims']
            latents: ArrayWithTime = sr.log['latents']

            for l in sr.stim_designer.log:
                s = l['s']
                v = l['v']
                proportion = proportion_in_space(v, s)
                proportions[-1][-1].append(proportion)

                stim_reg: BaseKernelRegressor = l['stim_reg']
                reg_i = l['observed_reg_input']
                reg_o = l['observed_s_hat']
                t_of_stim = l['time_of_stim']
                equivalent_projection_matrix = l['equiv_proj_mat']
                stim_sample = latents.time_to_sample(t_of_stim)
                old_v = latents[stim_sample-1] - latents[stim_sample-2]
                this_v = latents[stim_sample] - latents[stim_sample-1]
                old_v_direction = old_v / np.linalg.norm(old_v)


                if stim_reg is not None:
                    preq_error = np.linalg.norm(reg_o - stim_reg.predict(reg_i))
                else:
                    preq_error = np.nan
                preq_errors[-1][-1].append(preq_error)
                v_delta_errors[-1][-1].append(proportion_in_space(v, reg_o))
                s_delta_errors[-1][-1].append(np.linalg.norm(s - reg_o))
                angles[-1][-1].append(np.acos((reg_o / np.linalg.norm(reg_o)) @ (v / np.linalg.norm(v))))
                mags_along[-1][-1].append(reg_o @ (v / np.linalg.norm(v)))
                mags[-1][-1].append(np.linalg.norm(reg_o))
                alignment_with_old_v[-1][-1].append(np.acos((this_v / np.linalg.norm(this_v)) @ old_v_direction))
                v_mag_ratio[-1][-1].append(np.linalg.norm(this_v) / np.linalg.norm(old_v))

            # best_scale = stim_reg.cross_validate_length_scale(length_scales=np.logspace(-3,2, 20), depth=100)[0]
            # print(f"{k=} {best_scale=}")

            if preq_cutoff is not None:
                assert len(preq_errors[-1][-1]) >= preq_cutoff, f"to make the array non-ragged, we need to have at least {preq_cutoff} preq errors (not {len(preq_errors[-1][-1])})"
                preq_errors[-1][-1] = np.array(preq_errors[-1][-1][:preq_cutoff])
                v_delta_errors[-1][-1] = np.array(v_delta_errors[-1][-1][:preq_cutoff])
                s_delta_errors[-1][-1] = np.array(s_delta_errors[-1][-1][:preq_cutoff])
                angles[-1][-1] = np.array(angles[-1][-1][:preq_cutoff])
                mags_along[-1][-1] = np.array(mags_along[-1][-1][:preq_cutoff])
                mags[-1][-1] = np.array(mags[-1][-1][:preq_cutoff])
                alignment_with_old_v[-1][-1] = np.array(alignment_with_old_v[-1][-1][:preq_cutoff])
                v_mag_ratio[-1][-1] = np.array(v_mag_ratio[-1][-1][:preq_cutoff])
                proportions[-1][-1] = np.array(proportions[-1][-1][:preq_cutoff])

    if preq_cutoff is None:
        preq_cutoff = np.inf
        for a in preq_errors:
            for b in a:
                if len(b) < preq_cutoff:
                    preq_cutoff = len(b)
        for i in range(len(preq_errors)):
            for j in range(len(preq_errors[i])):
                preq_errors[i][j] = preq_errors[i][j][:preq_cutoff]
                v_delta_errors[i][j] = v_delta_errors[i][j][:preq_cutoff]
                s_delta_errors[i][j] = s_delta_errors[i][j][:preq_cutoff]
                angles[i][j] = angles[i][j][:preq_cutoff]
                mags_along[i][j] = mags_along[i][j][:preq_cutoff]
                mags[i][j] = mags[i][j][:preq_cutoff]
                alignment_with_old_v[i][j] = alignment_with_old_v[i][j][:preq_cutoff]
                v_mag_ratio[i][j] = v_mag_ratio[i][j][:preq_cutoff]
                proportions[i][j] = proportions[i][j][:preq_cutoff]
    return proportions, preq_errors, v_delta_errors, s_delta_errors, angles, mags_along, mags, alignment_with_old_v, v_mag_ratio


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

            proportions_new, preq_errors_new, v_delta_errors, s_delta_errors, angles, mags_along, mags, alignment_with_old_v, v_mag_ratio = extract_metrics(srs, preq_cutoff=50)
            proportions_original, preq_errors_original = extract_metrics_depreciated(srs, preq_cutoff=50)
            # TODO: this fails: `assert np.array_equal(proportions_new, proportions_original, equal_nan=True)`
            # getting rid of it will get rid of the depreciated call
            preq_errors_original = preq_errors_new  # assert np.array_equal(preq_errors_original, preq_errors_new, equal_nan=True) always passed

            fig, axs = plt.subplots(ncols=2, squeeze=False, figsize=(8,4), layout='constrained')
            to_plot = {k:v for k, v in zip(srs.keys(), [x[0] for x in proportions_original])}
            sns.violinplot(to_plot, orient='v', ax=axs[0,0])
            sns.swarmplot(to_plot, orient='v', ax=axs[0,0])

            for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors_original)):
                for j, e in enumerate(errors):
                    axs[0,1].plot(e, color=f'C{i}', alpha=0.1)
            for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors_original)):
                trendline = np.mean(errors, axis=0)
                axs[0,1].plot(trendline, color=f'C{i}', lw=1.5)
            axs[0, 1].semilogy()

        case 'optim_col_vs_rand_with_high_d_rand':
            d = datasets.Odoherty21Dataset()
            data = d.neural_data
            srs = make_srs(data=data, rng=rng, comparison_preset='optim_col_vs_rand_with_high_d_rand', n_runs=2, show_tqdm=True)

            proportions_new, preq_errors_new, v_delta_errors, s_delta_errors, angles, mags_along, mags, alignment_with_old_v, v_mag_ratio = extract_metrics(srs, preq_cutoff=50)
            proportions_original, preq_errors_original = extract_metrics_depreciated(srs, preq_cutoff=50)
            # TODO: this fails `assert np.array_equal(proportions_new, proportions_original, equal_nan=True)`
            preq_errors_original = preq_errors_new  # assert np.array_equal(preq_errors_original, preq_errors_new, equal_nan=True) always passed

            fig, axs = plt.subplots(ncols=4, nrows=2, squeeze=False, figsize=(15, 8), layout='constrained')

            to_plot = {k: np.array(v).flatten() * 180/np.pi for k, v in zip(srs.keys(), angles)}
            sns.violinplot(to_plot, orient='v', ax=axs[0, 0])
            sns.swarmplot(to_plot, orient='v', ax=axs[0, 0], size=3, edgecolor='white')
            axs[0,0].set_title('angle from desired vector ($Q_1$)')
            axs[0,0].set_ylabel('cosine angle (degrees)')

            print('mags along')
            to_plot = {k: np.array(v).flatten() for k, v in zip(srs.keys(), mags_along)}
            sns.violinplot(to_plot, orient='v', ax=axs[0, 1])
            sns.swarmplot(to_plot, orient='v', ax=axs[0, 1], size=3, edgecolor='white')
            for k in to_plot:
                print(f"\t{to_plot[k].mean() = } {to_plot[k].std() = }")
            axs[0,1].set_title('magnitude along desired vector ($Q_1$)')
            axs[0,1].set_ylabel('magnitude (a.u.)')

            print('mags')
            to_plot = {k: np.array(v).flatten() for k, v in zip(srs.keys(), mags)}
            sns.violinplot(to_plot, orient='v', ax=axs[0, 2])
            sns.swarmplot(to_plot, orient='v', ax=axs[0, 2], size=3, edgecolor='white')
            for k in to_plot:
                print(f"\t{to_plot[k].mean() = } {to_plot[k].std() = }")
            axs[0,2].set_title('total magnitude')
            axs[0,2].set_ylabel('magnitude (a.u.)')

            print('angle with old v')
            to_plot = {k: np.array(v).flatten() * 180/np.pi for k, v in zip(srs.keys(), alignment_with_old_v)}
            sns.violinplot(to_plot, orient='v', ax=axs[0, 3])
            sns.swarmplot(to_plot, orient='v', ax=axs[0, 3], size=3, edgecolor='white')
            for k in to_plot:
                print(f"\t{to_plot[k].mean() = } {to_plot[k].std() = }")
            axs[0,3].set_title('Angle with old direction of movement')
            axs[0,3].set_ylabel('angle')

            print('v_mag_ratio')
            to_plot = {k: np.array(v).flatten() for k, v in zip(srs.keys(), v_mag_ratio)}
            sns.violinplot(to_plot, orient='v', ax=axs[1, 0])
            sns.swarmplot(to_plot, orient='v', ax=axs[1, 0], size=3, edgecolor='white')
            for k in to_plot:
                print(f"\t{to_plot[k].mean() = } {to_plot[k].std() = }")
            axs[1,0].semilogy()
            axs[1,0].set_title('Step velocity ratio')
            axs[1,0].set_ylabel(r' $\log \frac{z_t - z_{t-1}}{z_{t-1} - z_{t-2}}$')

        case 'optim_open_vs_closed':
            data = datasets.Odoherty21Dataset().neural_data
            srs = make_srs(data=data, rng=rng, comparison_preset='optim_open_vs_closed', n_runs=N, show_tqdm=True, overrides=dict(last_dim_red=args.type_of_dim_red))

            proportions, preq_errors, v_delta_errors, s_delta_errors, angles, mags_along, mags, alignment_with_old_v, v_mag_ratio = extract_metrics(srs, preq_cutoff=None)
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
            proportions, preq_errors, v_delta_errors, s_delta_errors, angles, mags_along, mags, alignment_with_old_v, v_mag_ratio = extract_metrics(srs, preq_cutoff=None)
            fig = open_v_closed_plot(srs, proportions, preq_errors, v_delta_errors, s_delta_errors, show_individuals=False)
        case _:
            raise ValueError()


    fig.savefig(args.output, bbox_inches="tight")
