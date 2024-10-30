from adaptive_latents import (
    datasets,
    Concatenator,
    Pipeline,
    CenteringTransformer,
    KernelSmoother,
    proSVD,
    Bubblewrap,
    VanillaOnlineRegressor,
)
from adaptive_latents.timed_data_source import ArrayWithTime
from adaptive_latents.utils import evaluate_regression
import numpy as np
import timeit
from tqdm.auto import tqdm
import functools
import copy
import matplotlib.pyplot as plt


class PredictionRegressionRun:
    def __init__(
            self,
            neural_data,
            behavioral_data,
            dim_red_branches,
            targets,
            neural_lag = 0,
            neural_smoothing_tau=.12,
            stream_scaling_factors=None,
            exit_time=None,
    ):
        self.stream_scaling_factors = stream_scaling_factors or {0:1, 1:1}
        self.neural_smoothing_tau = neural_smoothing_tau
        self.neural_lag = neural_lag
        assert self.neural_lag >= 0

        neural_data = copy.deepcopy(neural_data)
        neural_data.t = neural_data.t + self.neural_lag

        streams = []
        streams.append((neural_data, 0))
        streams.append((behavioral_data, 1))
        streams.extend((value, key) for key, value in targets.items())

        exit_time = exit_time or max([source.t[-1] for source, stream in streams])

        initial_common_pipeline = Pipeline([
            CenteringTransformer(init_size=100),
            KernelSmoother(tau=neural_smoothing_tau / neural_data.dt),
            Concatenator(input_streams={0: 0, 1: 1}, output_streams={0: 2, 1: 2, 'skip': -1}, stream_scaling_factors=stream_scaling_factors)
        ])

        target_estimates = {dim_red_name: {target_name: [] for target_name in targets} for dim_red_name in dim_red_branches}

        # don't like this
        new_dim_red_branches = {}
        for k, (dim_red_method, bw, reg_constructor) in dim_red_branches.items():
            if bw is not None and reg_constructor is not None:
                new_dim_red_branches[k] = (dim_red_method, bw, {target_name: reg_constructor() for target_name in targets})
            else:
                new_dim_red_branches[k] = (dim_red_method, None, None)
        dim_red_branches = new_dim_red_branches

        dim_reduced_data = {name: [] for name in dim_red_branches}


        pbar = tqdm(total=exit_time)
        start_time = timeit.default_timer()
        for data, stream in initial_common_pipeline.streaming_run_on(streams, return_output_stream=True):
            if stream == 2:
                for dim_red_name, (dim_red_method, bw, regs) in dim_red_branches.items():
                    reduced_data = dim_red_method.partial_fit_transform(data, stream=0)
                    dim_reduced_data[dim_red_name].append(reduced_data)
                    if bw is not None:
                        bw:Bubblewrap
                        # update
                        alpha = np.array(bw.partial_fit_transform(data=reduced_data, stream=0))
                        for reg in regs.values():
                            reg.partial_fit_transform(alpha, stream=0)

                        # prediction part
                        if bw.is_initialized:
                            prediction_t = data.t + bw.dt
                            alpha_pred = bw.get_alpha_at_t(prediction_t)
                            for target_name in targets:
                                pred = regs[target_name].predict(alpha_pred)
                                target_estimates[dim_red_name][target_name].append(ArrayWithTime(pred, prediction_t))

            if stream in targets:
                for dim_red_name, (dim_red_method, bw, reg) in dim_red_branches.items():
                    if bw is not None:
                        reg[stream].partial_fit_transform(data, stream=1)

            pbar.update(round(data.t - pbar.n, 1))
            if data.t > exit_time:
                break

        self.computation_time = timeit.default_timer() - start_time

        self.targets = targets

        self.initial_common_pipeline = initial_common_pipeline
        self.dim_red_branches = {key: (dim_red_method, bw.uninitialized_copy() if bw is not None else bw, reg_constructor) for key, (dim_red_method, bw, reg_constructor) in dim_red_branches.items()}
        self.dim_reduced_data = {key: ArrayWithTime.from_list(value) for key, value in dim_reduced_data.items()}

        self.target_estimates = {}
        for dim_red_name, estimates in target_estimates.items():
            self.target_estimates[dim_red_name] = {}
            for target_name, estimate in estimates.items():
                self.target_estimates[dim_red_name][target_name] = ArrayWithTime.from_list(estimate)

        self.reg_performances = {}
        for dim_red_name, estimates in self.target_estimates.items():
            self.reg_performances[dim_red_name] = {}
            for target_name, estimate in estimates.items():
                self.reg_performances[dim_red_name][target_name] = {}
                estimate = self.target_estimates[dim_red_name][target_name]
                target = self.targets[target_name]
                corr, nrmse = evaluate_regression(estimate, estimate.t, target.a[:,0,:], target.t)
                self.reg_performances[dim_red_name][target_name]['corr'] = corr
                self.reg_performances[dim_red_name][target_name]['nrmse'] = np.array(nrmse)


    @classmethod
    def convenience_constructor(
            cls,
            n_bubbles=1100,
            bw_step=10**-1.5,
    ):
        # TODO: finish this to replicate the SFN graphs
        d = datasets.Odoherty21Dataset()
        bw = Bubblewrap(
            num=n_bubbles,
            M=500,
            lam=1e-3,
            nu=1e-3,
            eps=1e-4,
            step=bw_step,
            num_grad_q=1,
            sigma_orig_adjustment=100,
            log_level=1,
        )
        dim_red_branches = {
            'p6': [proSVD(k=6), bw, functools.partial(VanillaOnlineRegressor)],
            'p1': [proSVD(k=1), None, None]
        }
        targets = {
            'beh': d.behavioral_data
        }

        p = PredictionRegressionRun(d.neural_data, d.behavioral_data, dim_red_branches=dim_red_branches, targets=targets, exit_time=50)

        return p


    def plot_flow_fields(self, x_direction=0, y_direction=1, grid_n=13, scatter_alpha=0, normalize_method=None):
        assert normalize_method in {None, 'none', 'diffs', 'hcubes', 'squares'}
        fig, axs = plt.subplots(nrows=1, ncols=len(self.dim_red_branches), squeeze=False, layout='tight', figsize=(12,4))

        for idx, (name, latents) in enumerate(self.dim_reduced_data.items()):
            e1, e2 = np.zeros(latents.shape[1]), np.zeros(latents.shape[1])
            e1[0] = x_direction
            e2[1] = y_direction

            ax: plt.Axes = axs[0, idx]
            ax.scatter(latents @ e1, latents @ e2, s=5, alpha=scatter_alpha)
            x1, x2, y1, y2 = ax.axis()
            x_points = np.linspace(x1, x2, grid_n)
            y_points = np.linspace(y1, y2, grid_n)

            d_latents = np.diff(latents, axis=0)
            if normalize_method == 'diffs':
                d_latents = d_latents / np.linalg.norm(d_latents, axis=1)[:, np.newaxis]


            origins = []
            arrows = []
            n_points = []
            for i in range(len(x_points) - 1):
                for j in range(len(y_points) - 1):
                    proj_1 = (latents[:-1] @ e1)
                    proj_2 = (latents[:-1] @ e2)
                    # s stands for slice
                    s = (
                            (x_points[i] <= proj_1) & (proj_1 < x_points[i + 1])
                            &
                            (y_points[j] <= proj_2) & (proj_2 < y_points[j + 1])
                    )
                    if s.sum():
                        arrow = d_latents[s].mean(axis=0)
                        if normalize_method == 'hcubes':
                            arrow = arrow / np.linalg.norm(arrow)
                        arrows.append(arrow)
                        origins.append([x_points[i:i + 2].mean(), y_points[j:j + 2].mean()])
                        n_points.append(s.sum())
            origins, arrows, n_points = np.array(origins), np.array(arrows), np.array(n_points)
            arrows = np.array([arrows @ e1, arrows @ e2]).T
            if normalize_method == 'squares':
                arrows = arrows / np.linalg.norm(arrows, axis=1)[:, np.newaxis]

            ax.quiver(origins[:, 0], origins[:, 1], arrows[:,0], arrows[:,1], scale=1 / 20, units='dots', color='red')

            ax.axis('equal')
            ax.axis('off')

    @staticmethod
    def compare_metrics_across_runs(tried, runs, fig=None, axs=None):
        tried, runs = zip(*sorted(zip(tried, runs)))
        for run in runs:
            assert len(run.dim_red_branches) == 1
            assert run.dim_red_branches[list(run.dim_red_branches.keys())[0]][1] is not None

        targets = list(runs[0].targets.keys())
        dim_red_name = list(runs[0].dim_red_branches.keys())[0]

        if fig is None:
            fig, axs = plt.subplots(nrows=len(targets), ncols=2, squeeze=False, figsize=(4*2,5/3 * len(targets)), sharex=False)
        else:
            for ax in axs.flatten():
                ax.cla()

        for idx, target_str in enumerate(targets):
            run: PredictionRegressionRun
            correlations = np.array([run.reg_performances[dim_red_name][target_str]['corr'] for run in runs])
            mses = np.array([run.reg_performances[dim_red_name][target_str]['nrmse'] for run in runs])
            for jdx, metric in enumerate([correlations, mses]):
                axs[idx,jdx].plot(tried, metric)
                metric = -metric if jdx == 1 else metric
                best_tried = tried[np.argmax(metric.sum(axis=1))]
                axs[idx, jdx].axvline(best_tried, color='k')
                axs[idx, jdx].text(.99, .99, f'{best_tried:.3f}', ha='right', va='top', transform=axs[idx, jdx].transAxes)
                if idx != 2:
                    axs[idx, jdx].set_xticks(tried)
                    axs[idx, jdx].set_xticklabels([])
            axs[idx,0].set_ylabel(target_str)
        axs[0,0].set_title('correlation')
        axs[0,1].set_title('NRMSE')
        return fig, axs
