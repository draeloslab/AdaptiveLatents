import numpy as np
import matplotlib.pyplot as plt
import adaptive_latents
import itertools
from collections import deque
from adaptive_latents import CenteringTransformer, Pipeline, proSVD, KernelSmoother, sjPCA, mmICA
from adaptive_latents.bw_run import AnimationManager
from datasets import Leventhal24uDataset
from tqdm import tqdm

if __name__ == '__main__':
    d = Leventhal24uDataset(bin_size=.1)

    pipelines = [
        Pipeline([
            KernelSmoother(tau=8),
            CenteringTransformer(),
            proSVD(k=6),
            KernelSmoother(tau=8),
            sjPCA()
        ]),
        # Pipeline([
        #     KernelSmoother(tau=8),
        #     CenteringTransformer(),
        #     proSVD(k=6, whiten=True),
        #     KernelSmoother(tau=8),
        #     mmICA()
        # ]),
    ]

    start = 200 # in s
    video_time = 3500-start # in s
    tq = tqdm(total=start+video_time)

    with AnimationManager(fps=40, n_rows=2, n_cols=2) as am:
        traj_ax: plt.Axes = am.ax[0, 0]
        timing_ax: plt.Axes = am.ax[1, 0]

        outputs = [[] for p in pipelines]
        for i, output_rows in enumerate(zip(*[p.run_on(d.neural_data) for p in pipelines])):
            if i < 100:
                continue

            for j, row in enumerate(output_rows):
                outputs[j].append(row)
            os = [np.squeeze(o) for o in outputs]

            current_ts = [p.mid_run_sources[0][0].current_sample_time() for p in pipelines]
            assert all(t == current_ts[0] for t in current_ts)
            current_t = current_ts[0]
            o = os[0]
            if len(o.shape) == 2 and start < current_t < start + video_time:
                traj_ax.cla()
                traj_ax.autoscale(True)

                for j, o in enumerate(os):
                    traj_ax.scatter(o[:, 0], o[:, 1], alpha=.999 ** np.arange(o.shape[0])[::-1], color=f'C{j}', linewidth=0, s=5)
                    traj_ax.plot(o[-10:, 0], o[-10:, 1], color=f'C{j}')
                traj_ax.set_title(f't= {current_t:.1f}s', fontfamily='monospace')
                traj_ax.axis('equal')


                timing_ax.cla()
                timing_ax.axvline(0, color='k')
                # 'foodClick',
                for i, key in enumerate(['cueOn', 'centerIn', 'tone', 'centerOut', 'sideIn', 'sideOut',  'foodRetrieval']):
                    near_events = d.trial_data[key] - current_t
                    near_events = near_events[np.abs(near_events) < 50]
                    for j, event in enumerate(near_events):
                        timing_ax.axvline(event, color=f'C{i}', label=key if j == 0 else "")
                    timing_ax.set_xlim([-2,4])
                timing_ax.legend(loc='upper right')

                am.grab_frame()
            elif current_t > start + video_time:
                break
            tq.update(np.floor(current_t - tq.n).astype(int))
