import numpy as np

import adaptive_latents
import seaborn as sns
from sim_stim import make_srs, make_slices_tensor
from adaptive_latents import datasets
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
    d = datasets.Odoherty21Dataset()
    data = d.neural_data

    match args.type_of_plot:
        case 'optim_col_vs_rand':
            preq_cutoff = 50
            srs = make_srs(data=data, rng=rng, comparison_preset='optim_col_vs_rand', n_runs=10, show_tqdm=True)
            proportions = []
            preq_errors = []
            for k, sr_list in srs.items():
                preq_errors.append([])
                proportions.append([])
                for sr in sr_list:
                    preq_errors[-1].append([])

                    old_stim_reg = None
                    for l in sr.stim_designer.log:
                        s = l['s']
                        v = l['v']
                        proportion = proportion_in_space(v, s)
                        proportions[-1].append(proportion)

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


            fig, axs = plt.subplots(ncols=2, squeeze=False, layout='constrained')

            to_plot = {k:v for k, v in zip(srs.keys(), proportions)}
            sns.violinplot(to_plot, orient='h', ax=axs[0,0])
            sns.swarmplot(to_plot, orient='h', ax=axs[0,0])

            for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
                for j, e in enumerate(errors):
                    axs[0,1].plot(e, color=f'C{i}', alpha=0.1)

            for i, (k, errors) in enumerate(zip(srs.keys(), preq_errors)):
                trendline = np.mean(errors, axis=0)
                axs[0,1].plot(trendline, color=f'C{i}', lw=1.5)

        case _:
            raise ValueError()


    fig.savefig(args.output, bbox_inches="tight")
