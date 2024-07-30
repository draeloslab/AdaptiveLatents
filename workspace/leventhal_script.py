import numpy as np
import matplotlib.pyplot as plt
import adaptive_latents
from adaptive_latents import CenteringTransformer, Pipeline, proSVD, KernelSmoother, sjPCA
from datasets import Leventhal24uDataset

if __name__ == '__main__':
    d = Leventhal24uDataset(bin_size=.1)

    pro = proSVD(k=2, init_size=10, log_level=0)
    p = Pipeline([
        KernelSmoother(kernel_length=4, tau=1),
        CenteringTransformer(),
        pro,
        sjPCA()
    ])

    output = p.offline_run_on(d.neural_data, convinient_return=True)
    fig, ax = plt.subplots()

    # output, t = adaptive_latents.utils.clip(output[:,0], d.neural_data.t)
    # ax.plot(t, output)
    # for trial_start_time in d.trial_data['Time']:
    #     ax.axvline(trial_start_time, color='k')
    # ax.set_title('Trial start times')
    # ax.set_xlabel('time (s)')
    # ax.set_ylabel('proSVD output 1')

    # plt.hist2d(output[100:, 0], output[100:, 1], bins=20)
    plt.scatter(output[100:, 0], output[100:, 1], alpha=.1)
    plt.tight_layout()
    plt.axis('equal')

    plt.show()
