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


if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    parser.add_argument( "--type-of-plot", type=str, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(0)

    match args.type_of_plot:
        case 'optim_col_vs_rand':
            d = datasets.Odoherty21Dataset()
            data = d.neural_data

            preq_cutoff = 50
            srs = make_srs(data=data, rng=rng, comparison_preset='optim_col_vs_rand', n_runs=10, show_tqdm=True)
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
                            most_recent_row = stim_reg.history[stim_reg.n_observed - 1, :]
                            i = most_recent_row[:stim_reg.input_d]
                            o = most_recent_row[stim_reg.input_d:]

                            if old_stim_reg.history is None:
                                preq_error = np.nan
                            else:
                                preq_error = np.linalg.norm(o - old_stim_reg.predict(i))
                            preq_errors[-1][-1].append(preq_error)

                        old_stim_reg = stim_reg
                    assert len(preq_errors[-1][-1]) >= preq_cutoff, f"to make the array non-ragged, we need to have at least {preq_cutoff} preq errors (not {len(preq_errors[-1][-1])})"
                    preq_errors[-1][-1] = np.array(preq_errors[-1][-1][:preq_cutoff])


            fig, axs = plt.subplots(ncols=2, squeeze=False, figsize=(8,4), layout='constrained')

            # to_plot = {k:v for k, v in zip(srs.keys(), [np.hstack(x) for x in proportions])}
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
        case 'optim_open_vs_closed':
            d = datasets.Odoherty21Dataset()
            data = d.neural_data

            preq_cutoff = None
            srs = make_srs(data=data, rng=rng, comparison_preset='optim_open_vs_closed', n_runs=2, show_tqdm=True)
            proportions = []
            preq_errors = []
            v_delta_errors = []
            s_delta_errors = []
            for k, sr_list in srs.items():
                preq_errors.append([])
                v_delta_errors.append([])
                s_delta_errors.append([])
                proportions.append([])
                for sr in sr_list:
                    preq_errors[-1].append([])
                    v_delta_errors[-1].append([])
                    s_delta_errors[-1].append([])
                    proportions[-1].append([])

                    for l in sr.stim_designer.log:
                        s = l['s']
                        v = l['v']
                        proportion = proportion_in_space(v, s)
                        proportions[-1][-1].append(proportion)

                        stim_reg: BaseKernelRegressor = l['stim_reg']
                        reg_i = l['observed_reg_inpt']
                        reg_o = l['observed_s_hat']

                        if stim_reg is not None:
                            preq_error = np.linalg.norm(reg_o - stim_reg.predict(reg_i))
                        else:
                            preq_error = np.nan
                        preq_errors[-1][-1].append(preq_error)
                        v_delta_errors[-1][-1].append(proportion_in_space(v, reg_o))
                        s_delta_errors[-1][-1].append(np.linalg.norm(s - reg_o))

                    best_scale = stim_reg.cross_validate_length_scale(length_scales=np.logspace(-3,2, 20), depth=100)[0]
                    # print(f"{k=} {best_scale=}")

                    if preq_cutoff is not None:
                        assert len(preq_errors[-1][-1]) >= preq_cutoff, f"to make the array non-ragged, we need to have at least {preq_cutoff} preq errors (not {len(preq_errors[-1][-1])})"
                        preq_errors[-1][-1] = np.array(preq_errors[-1][-1][:preq_cutoff])
                        v_delta_errors[-1][-1] = np.array(v_delta_errors[-1][-1][:preq_cutoff])
                        s_delta_errors[-1][-1] = np.array(s_delta_errors[-1][-1][:preq_cutoff])
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
                        proportions[i][j] = proportions[i][j][:preq_cutoff]


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
            for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
                for j, e in enumerate(errors):
                    ax.plot(e, color=f'C{i}', alpha=0.1)

            for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
                trendline = np.mean(errors, axis=0)
                ax.plot(trendline, color=f'C{i}', lw=1.5)
            ax.set_title('$\\Vert \\hat s_n - \\hat S_{n-1}(x_n, u_n) \\Vert$')
            ax.semilogy()

        case _:
            raise ValueError()


    fig.savefig(args.output, bbox_inches="tight")
