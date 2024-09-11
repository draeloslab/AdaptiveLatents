import numpy as np
import matplotlib.pyplot as plt
import adaptive_latents
import itertools
from collections import deque
from adaptive_latents import CenteringTransformer, Pipeline, proSVD, KernelSmoother, sjPCA, mmICA, AnimationManager
from adaptive_latents import datasets
from tqdm import tqdm

def make_video(outdir=None):
    "shows a video of three pairs of latents with events in the bottom-left corner"
    d = datasets.Leventhal24uDataset(bin_size=.1)

    smoother1 = KernelSmoother(tau=8)
    centerer = CenteringTransformer()
    pro = proSVD(k=6, whiten=False)
    smoother2 = KernelSmoother(tau=8)
    jpca = sjPCA()
    # mmica = mmICA()

    p = Pipeline([
        smoother1,
        centerer,
        pro,
        smoother2,
        jpca,
    ])

    # start = 665 # didn't go
    start = 1106
    video_time = 10
    # video_time = 3500-start # in s
    tq = tqdm(total=start+video_time)

    with AnimationManager(fps=40, n_rows=2, n_cols=2, outdir=outdir) as am:
        traj_ax: plt.Axes = am.axs[0, 0]
        timing_ax: plt.Axes = am.axs[1, 0]

        outputs = [[],[],[],[],[],[]]
        for row in d.neural_data:
            p._partial_fit(row, stream=0)
            current_time = d.neural_data.current_sample_time()

            output = p.transform(row, stream=0)
            for i in range(len(outputs)):
                outputs[i].append(output[0][i])

            if start <= current_time < start + video_time:
                # plotting
                traj_ax.cla()

                # center = pro.transform(centerer.center[None, :])[0]
                # adaptive_latents.plotting_functions.add_2d_bubble(traj_ax, pro.get_cov_matrix(), center, passed_sig=True, alpha=.1, n_sds=1)


                i1, i2 = 0, 1
                am.axs[0,0].cla()
                am.axs[0,0].scatter(outputs[i1], outputs[i2], alpha=.999 ** np.arange(len(outputs[0]))[::-1], linewidth=0, s=5)
                am.axs[0,0].plot(outputs[i1][-10:], outputs[i2][-10:])
                am.axs[0, 0].axis('equal')
                am.axs[0, 0].set_title(f'latent {i1} vs latent {i2}')

                i1, i2 = 2, 3
                am.axs[0,1].cla()
                am.axs[0, 1].scatter(outputs[i1], outputs[i2], alpha=.999 ** np.arange(len(outputs[0]))[::-1], linewidth=0, s=5)
                am.axs[0, 1].plot(outputs[i1][-10:], outputs[i2][-10:])
                am.axs[0, 1].axis('equal')
                am.axs[0, 1].set_title(f'latent {i1} vs latent {i2}')

                i1, i2 = 4, 5
                am.axs[1,1].cla()
                am.axs[1,1].scatter(outputs[i1], outputs[i2], alpha=.999 ** np.arange(len(outputs[0]))[::-1], linewidth=0, s=5)
                am.axs[1,1].plot(outputs[i1][-10:], outputs[i2][-10:])
                am.axs[1, 1].axis('equal')
                am.axs[1, 1].set_title(f'latent {i1} vs latent {i2}')


                timing_ax.cla()
                timing_ax.axvline(0, color='k')
                for i, key in enumerate(['cueOn', 'centerIn', 'tone', 'centerOut', 'sideIn', 'sideOut', 'foodRetrieval']):
                    near_events = d.trial_data[key] - current_time
                    near_events = near_events[np.abs(near_events) < 50]
                    for j, event in enumerate(near_events):
                        timing_ax.axvline(event, color=f'C{i}', label=key if j == 0 else "")
                    timing_ax.set_xlim([-2, 4])
                timing_ax.legend(loc='upper right')
                timing_ax.set_title(f't= {current_time:.1f}s', fontfamily='monospace')

                am.grab_frame()
            elif current_time > start + video_time:
                break

            tq.update(round(current_time,1)-tq.n)

def show_events_timestamps_and_average_trace(show=True):
    d = datasets.Leventhal24uDataset(bin_size=.1)

    smoother1 = KernelSmoother(tau=8)
    centerer = CenteringTransformer()
    pro = proSVD(k=6, whiten=False)
    smoother2 = KernelSmoother(tau=8)
    jpca = sjPCA()

    p = Pipeline([
        smoother1,
        # centerer,
        # pro,
        # smoother2,
        # jpca,
    ])

    output = p.offline_run_on(d.neural_data, convinient_return=True)
    t = d.neural_data.t
    output, t = adaptive_latents.utils.clip(output,t)

    fix, ax = plt.subplots()
    for i, key in enumerate(['cueOn', 'centerIn', 'tone', 'centerOut', 'sideIn', 'sideOut', 'foodRetrieval']):
        for j, event in enumerate(d.trial_data[key]):
            ax.axvline(event, color=f'C{i}', label=key if j == 0 else "")

    ax.plot(t, output.mean(axis=1))
    if show:
        plt.show()


def show_response_arcs(show=True):
    "shows the arcs of responses in the latent space"
    d = datasets.Leventhal24uDataset(bin_size=.1)

    centerer = CenteringTransformer()
    pro = proSVD(k=6, whiten=False)
    jpca = sjPCA()

    p = Pipeline([
        KernelSmoother(tau=8),
        centerer,
        pro,
        KernelSmoother(tau=8),
        jpca,
    ])
    p.offline_run_on(d.neural_data)



    centerer.freeze()
    pro.freeze()
    jpca.freeze()
    p = Pipeline([
        KernelSmoother(tau=8),
        centerer,
        pro,
        KernelSmoother(tau=8),
        jpca,
    ])
    output = p.offline_run_on(d.neural_data, convinient_return=True)
    t = d.neural_data.t
    output, t = adaptive_latents.utils.clip(output,t)

    peri_response_window = [-100, 0]
    event_index = 1
    trial_event_sequence = ['cueOn', 'centerIn', 'tone', 'centerOut', 'sideIn', 'sideOut', 'foodRetrieval']

    responses = []
    for event in d.trial_data[trial_event_sequence[event_index]]:
        if not np.isnan(event):
            rel_t = t - event
            near_events = (d.trial_data[trial_event_sequence] - event)
            next_event = near_events[near_events > 0].min(axis=None)
            last_event = near_events[near_events < 0].max(axis=None)
            peri_response_window[0] = max(peri_response_window[0], last_event)
            peri_response_window[1] = min(peri_response_window[1], next_event)
            selector = (peri_response_window[0] < rel_t) & (rel_t < peri_response_window[1])
            responses.append(output[selector, :])
            rel_t = rel_t[selector]


    fig, axs = plt.subplots(nrows=2, ncols=2)
    for i, j, idx1, idx2 in [
        (0, 0, 0, 1),
        (0, 1, 2, 3),
        (1, 1, 4, 5),
    ]:
        axs[i,j].scatter(output[:,idx1], output[:,idx2], alpha=.05, s=5, color='k', linewidth=0)
        for response in responses:
            axs[i,j].plot(response[:,idx1], response[:,idx2], color=f'C{event_index}', linewidth=.5, alpha=1)
    if show:
        plt.show()


if __name__ == '__main__':
    make_video()
