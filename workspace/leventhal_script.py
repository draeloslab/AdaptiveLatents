import numpy as np
import matplotlib.pyplot as plt
import adaptive_latents
import itertools
from adaptive_latents import CenteringTransformer, Pipeline, proSVD, KernelSmoother, sjPCA, mmICA
from adaptive_latents.bw_run import AnimationManager
from datasets import Leventhal24uDataset

if __name__ == '__main__':
    d = Leventhal24uDataset(bin_size=.1)

    p = Pipeline([
        KernelSmoother(kernel_length=10, tau=2),
        CenteringTransformer(),
        proSVD(k=6, init_size=10, whiten=False, log_level=0),
    ])

    with AnimationManager(fps=40) as am:
        output = []
        for i, output_row in enumerate(p.run_on(d.neural_data, return_output_stream=False)):
            if np.isnan(output_row).any():
                continue
            output.append(output_row)
            o = np.squeeze(output)

            if len(o.shape) == 2 and 500 < i < 1000:
                am.ax[0,0].cla()
                am.ax[0,0].scatter(o[:,0], o[:,1], alpha=.99**np.arange(o.shape[0])[::-1], color='C0')
                am.ax[0,0].plot(o[-10:,0], o[-10:,1], color='C1')

                am.grab_frame()


    # output = p.offline_run_on(d.neural_data, convinient_return=True)
    # fig, ax = plt.subplots()
    # plt.scatter(output[100:, 0], output[100:, 1], alpha=.1)
    # plt.plot(output[-100:, 0], output[-100:, 1], color='C1')
    # plt.tight_layout()
    # plt.axis('equal')
    # plt.show()

    # plt.plot(d.neural_data.a[:,0,0])
    # plt.plot(output[:,0])
    # plt.show()
