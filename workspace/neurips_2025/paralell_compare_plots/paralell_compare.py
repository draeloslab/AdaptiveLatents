import importlib
import numpy as np
import matplotlib.pyplot as plt

from adaptive_latents import datasets, proSVD, Pipeline, CenteringEstimator, StreamingKalmanFilter, Bubblewrap, sjPCA, mmICA, ArrayWithTime, plotting_functions, KernelSmoother, VJF
from adaptive_latents.utils import save_to_cache
from tqdm.auto import tqdm
from IPython import display
from adaptive_latents.plotting_functions import plot_flow_fields, AnimationManager, plot_history_with_tail
import adaptive_latents
import importlib

from adaptive_latents.predictor import Predictor
from adaptive_latents.regressions import BaseKernelRegressor


def main():
    d = datasets.Zong22Dataset()


    prosvd_k = 8

    @save_to_cache("parallel_compare")
    def f():
        p = Pipeline([CenteringEstimator(), KernelSmoother(tau=2 * .68 / d.neural_data.dt), proSVD(k=prosvd_k)])

        dim_red_methods = [Pipeline(), sjPCA(), mmICA()]
        # predictors = [StreamingKalmanFilter(log_level=2, check_dt=True, n_steps_to_predict=1, steps_between_refits=50) for _ in dim_red_methods]
        predictors = [Bubblewrap(log_level=2, check_dt=True, n_steps_to_predict=1) for _ in dim_red_methods]
        # predictors = [VJF(log_level=2, check_dt=True, n_steps_to_predict=1) for _ in dim_red_methods]

        regs = [BaseKernelRegressor(maxlen=10000, length_scale=0.1725) for _ in dim_red_methods]

        outputs = [[] for _ in dim_red_methods]

        pbar = tqdm(total=round(d.neural_data.t.max(),2))
        for data in p.streaming_run_on(d.neural_data):

            metrics = []
            in_space_data = []
            for dim_red_method, predictor, output_accumulator in zip(dim_red_methods, predictors, outputs):
                in_space_datum = dim_red_method.partial_fit_transform(data)
                in_space_datum = in_space_datum[:,:4]
                in_space_data.append(in_space_datum)
                output_accumulator.append(in_space_datum)

                mse = ((in_space_datum - predictor.predict(1)) ** 2).mean()
                neg_log_pred_p = -predictor.unevaluated_log_pred_p(1)(in_space_datum)
                metrics.append(neg_log_pred_p)
                predictor.partial_fit_transform(in_space_datum)

            best_regressor = np.argmin(metrics)
            for i, (reg, in_space_datum) in enumerate(zip(regs, in_space_data)):
                reg.observe(in_space_datum, np.array([i == best_regressor]))

            pbar.update(round(data.t,2) - pbar.n)


        outputs = [ArrayWithTime.from_list(o, drop_early_nans=True, squeeze_type='to_2d') for o in outputs]
        return outputs, dim_red_methods, regs

    outputs, dim_red_methods, regs = f()



    from scipy.signal import convolve2d
    from scipy.ndimage import gaussian_filter

    labels = ['prosvd','sjpca','mmica']

    def make_heatmap(ax, o, x_direction, y_direction, color_direction, density=13, limits=None, sigma=1, cax=None):
        e1, e2, ec = np.zeros(o.shape[1]), np.zeros(o.shape[1]), np.zeros(o.shape[1])
        e1[x_direction] = 1
        e2[y_direction] = 1
        ec[color_direction] = 1
        x = o @ e1
        y = o @ e2
        c = o @ ec

        # ax.scatter(x, y, c=c, s=1, cmap='plasma')

        if limits is None:
            axis = ax.axis()
        else:
            axis = limits

        x_edges = np.linspace(axis[0], axis[1], density + 1)
        y_edges = np.linspace(axis[2], axis[3], density + 1)
        x_centers = np.convolve(x_edges, [0.5, 0.5], mode='valid')
        y_centers = np.convolve(y_edges, [0.5, 0.5], mode='valid')

        x_grid, y_grid = np.meshgrid(x_centers, y_centers)
        c_grid = np.zeros_like(x_grid)

        for i in range(len(y_centers)):
            for j in range(len(x_centers)):
                slice_1 = (x_edges[j] < x) & (x < x_edges[j+1])
                slice_2 = (y_edges[i] < y) & (y < y_edges[i+1])
                s = (slice_1 & slice_2)
                if s.sum()<1:
                    c_grid[i,j] = 0
                else:
                    c_grid[i,j] = np.mean(c[s])

        c_grid = gaussian_filter(c_grid,sigma=sigma, mode='constant', cval=0.0)

        cmap = plt.colormaps['plasma']
        cmap.set_bad('k')
        cmesh = ax.pcolormesh(x_grid, y_grid, c_grid, cmap=cmap, vmin=0, vmax=.58)
        print(f'{c_grid.min()=:.2f} {c_grid.max()=:.2f}')
        if cax is not None:
            fig.colorbar(cmesh, cax=cax, orientation='horizontal')



    fig, axs = plt.subplots(3, len(dim_red_methods), figsize=(10, 5), squeeze=False, sharex='col', sharey='col', layout='constrained', height_ratios=[.1, 1,1])
    gs = axs[0,0].get_gridspec()

    for ax in axs[0,:]:
        ax.remove()
    cax = fig.add_subplot(gs[0,:])

    x_direction = 0
    y_direction = 1

    mmica_limits = [-5, 5.1, -6.7, 8]
    plot_limits = [None, None, mmica_limits]
    for k,v,limit,ax,arrow_scale in zip(labels, outputs, plot_limits, axs[1], [.75,1,1]):
        adaptive_latents.plotting_functions.plot_flow_fields(
            {k:v},
            # method='streamplot',
            method='quiver',
            grid_n=20,
            # normalize_method='diffs',
            # normalize_method='none',
            normalize_method='diffs',
            # normalize_method='hcubes',
            fig=fig, axs=[ax], format_axis=False,
            x_direction=x_direction, y_direction=y_direction, scatter_alpha=0,
            limits=limit,
            f_on_arrows=lambda x: x * arrow_scale *.8,
        )



    for reg, ax, l, o in zip(regs, axs[1], plot_limits, outputs):
        # ax.scatter(reg.history[:,x_direction], reg.history[:,y_direction], c=reg.history[:,-1], s=1, cmap='plasma', vmin=-.1, vmax=1.1)

        stride = 10
        base = 17 * stride
        to_plot = o.slice_by_time(slice(base, base+stride+2.5))
        line = ax.plot(to_plot[:,x_direction], to_plot[:,y_direction], 'k', lw=1.25)
        # ax.scatter(reg.history[:,x_direction], reg.history[:,y_direction], c=reg.history[:,-1], s=1, cmap='plasma', vmin=-.1, vmax=1.1)
        h = reg.history[o.time_to_sample(to_plot.t), :]
        # ax.scatter(h[:,x_direction], h[:,y_direction], c=h[:,-1], cmap='plasma')

        for arrow_index in [45]:
            ax.annotate('',
                             xytext=(to_plot[arrow_index, x_direction], to_plot[arrow_index, y_direction]),
                             xy=(to_plot[arrow_index+1, x_direction], to_plot[arrow_index+1, y_direction]),
                             arrowprops=dict(arrowstyle="simple", color='k'),
                             size=11
                             )

        ax.axis(l)



    for reg, ax, old_ax in zip(regs, axs[2], axs[1]):
        density = 200
        make_heatmap(ax, reg.history, x_direction, y_direction, color_direction=-1, density=density, sigma=density * 4/200, cax=cax)

    return fig

if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    fig = main()

    fig.savefig(args.output, bbox_inches="tight")

