import numpy as np
from adaptive_latents import datasets
from sim_stim import make_srs
import pathlib
import matplotlib.pyplot as plt


def main():
    rng = np.random.default_rng(0)
    d = datasets.Zong22Dataset()
    data = d.neural_data

    srs = make_srs(data, rng, comparison_preset='visualization', n_runs=1, show_tqdm=True)


    i= 40
    sr = srs['learning from stim'][0]

    fig, axs = plt.subplots(ncols=2, figsize=(10,4), sharex=False, sharey=False, layout='constrained')

    latents = sr.log['latents'].slice_by_time(slice(30,None))
    axs[0].plot(latents[:, 0], latents[:, 1], alpha=.1, color='k')
    stim_s = sr.log['stim_intended_samples'].t - latents.dt

    l = 1
    r = 5.1
    ax_n = 0
    center_t = sr.log['stim_intended_samples'].t[i]
    latents = sr.log['latents'].slice_by_time(slice(center_t-l,center_t+r))
    line = axs[ax_n].plot(latents[:, 0], latents[:, 1])
    stim_s = sr.log['stim_intended_samples'].slice_by_time(slice(center_t-l,center_t+r)).t - latents.dt
    latents_s = latents.slice_by_time(stim_s).reshape((-1, latents.shape[1]))
    axs[ax_n].plot(latents_s[:, 0], latents_s[:, 1], '.', color='r')

    for arrow_index in [17, 50]:
        axs[0].annotate('',
                         xytext=(latents[arrow_index, 0], latents[arrow_index, 1]),
                         xy=(latents[arrow_index+1, 0], latents[arrow_index+1, 1]),
                         arrowprops=dict(arrowstyle="simple", color='C0'),
                         size=11
                         )


    u = sr.stim_designer.log[i]['u']
    idx = np.argsort(np.abs(u))[::-1]
    print(np.linalg.norm(u,ord=0))

    high_d = sr.log['high_d_with_stim'].slice_by_time(slice(center_t-l,center_t+r))
    axs[1].plot(high_d.t, high_d[:,idx[:int(np.linalg.norm(u,ord=0))]]);
    for stim_t in stim_s:
        axs[1].axvline(stim_t, color='r')

    return fig


if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    fig = main()


    fig.savefig(args.output, bbox_inches="tight")
