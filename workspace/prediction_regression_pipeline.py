import adaptive_latents
from adaptive_latents import (
    AnimationManager,
    Pipeline,
    CenteringTransformer,
    proSVD,
    Bubblewrap,
    VanillaOnlineRegressor,
    NumpyTimedDataSource,
    KernelSmoother,
    Concatenator,
    sjPCA,
    ZScoringTransformer,
    mmICA
)
from tqdm.notebook import tqdm
from adaptive_latents.timed_data_source import ArrayWithTime
from adaptive_latents.utils import resample_matched_timeseries
from adaptive_latents import datasets
import numpy as np
import timeit
import matplotlib.pyplot as plt
from IPython.display import Video, Image, display
import socket



class PipelineRun:
    def __init__(
            self,
            neural_lag=0,
            neural_smoothing_tau=.12,
            concat_zero_streams=(),
            latents_for_bw='jpca',
            pre_bw_latent_dims_to_drop=0,
            pos_rescale_factor=1,
            vel_rescale_factor=1,
            alpha_pred_method='normal',
            bw_step=10**-1.5,
            n_bubbles=1100,
            shortrun=None,
    ):
        """
        Parameters
        ----------
        neural_lag
            optimized to 0, 1.7 is also interesting
        neural_smoothing_tau
            optimized to 0.12, should try again with behavior small
        pos_rescale_factor, vel_rescale_factor
            optimized to 1, but performance could increase with higer values
            1/30, 1/75 is also a local minimum
        bw_step
            optimized to 10**-1.5, but it's pretty flat, we could go bigger maybe


        latents_for_bw: {'prosvd', 'jpca', 'mmica'}
        """

        if shortrun is None:
            shortrun = socket.gethostname() == 'tycho' and True

        d = datasets.Odoherty21Dataset(neural_lag=neural_lag, pos_rescale_factor=pos_rescale_factor, vel_rescale_factor=vel_rescale_factor)

        streams = []
        streams.append((d.neural_data, 0))
        streams.append((d.beh_pos, 1))
        # 2 is reserved for the post-concatination pipeline
        streams.append((d.beh_pos, 3))
        streams.append((d.neural_data, 4))
        # 5 for the alpha to joint

        # this pipeline makes the latent space
        p1 = Pipeline([
            neural_centerer := CenteringTransformer(init_size=100, input_streams={0: 'X'}, output_streams={0: 0}),
            # CenteringTransformer(init_size=100, input_streams={1: 'X'}, output_streams={1: 1}),
            nerual_smoother := KernelSmoother(tau=neural_smoothing_tau / d.bin_width, input_streams={0: 'X'}, output_streams={0: 0}),
            Concatenator(input_streams={0: 0, 1: 1}, output_streams={0: 2, 1: 2, 'skip': -1}, zero_sreams=concat_zero_streams),
        ])

        pro = proSVD(k=6 + pre_bw_latent_dims_to_drop, init_size=100, input_streams={2: 'X'}, output_streams={2: 2})
        jpca = sjPCA(init_size=100, input_streams={2: 'X'}, output_streams={2: 2})
        ica = mmICA(init_size=100, input_streams={2: 'X'}, output_streams={2: 2})


        # this pipeline handles the prediction and regression
        bw = Bubblewrap(
            num=n_bubbles,
            M=500,
            lam=1e-3,
            nu=1e-3,
            eps=1e-4,
            step=bw_step,
            num_grad_q=1,
            sigma_orig_adjustment=100,  # 0
            input_streams={2: 'X'},
            output_streams={2: 2},
            log_level=1,
        )

        alpha_to_beh_reg = VanillaOnlineRegressor(
            input_streams={2: 'X', 3: 'Y'},
        )

        neural_only_reg_pipeline = Pipeline(
        [
                CenteringTransformer(**(neural_centerer.get_params() | dict(input_streams={4: 'X'}, output_streams={4: 4}))),
                KernelSmoother(**(nerual_smoother.get_params() | dict(input_streams={4: 'X'}, output_streams={4: 4}))),
                proSVD(k=4, init_size=100, input_streams={4: 'X'}, output_streams={4: 4}),
                alpha_to_neural_reg := VanillaOnlineRegressor(input_streams={2: 'X', 4: 'Y'})
            ],
            reroute_inputs=False
        )

        alpha_to_joint_latents_reg = VanillaOnlineRegressor(
            input_streams={2: 'X', 5: 'Y'},
        )

        exit_time = 40 if shortrun else 600

        pbar = tqdm(total=exit_time)
        pro_latents = []
        jpca_latents = []
        ica_latents = []

        next_bubble_predictions = []
        beh_predictions = []
        neural_predictions = []
        joint_predictions = []

        beh_target = []
        neural_target = []
        joint_target = []

        start_time = timeit.default_timer()
        for output, stream in p1.streaming_run_on(streams, return_output_stream=True):
            # prosvd step
            pro_output, pro_stream = pro.partial_fit_transform(output, stream, return_output_stream=True)
            if pro_stream == 2 and np.isfinite(pro_output).all():
                pro_latents.append(pro_output)

            # sjpca step
            jpca_output, jpca_stream = jpca.partial_fit_transform(pro_output, pro_stream, return_output_stream=True)
            if jpca_stream == 2 and np.isfinite(jpca_output).all():
                jpca_latents.append(jpca_output)

            # ica step (not in main line)
            ica_output, ica_stream = ica.partial_fit_transform(pro_output, pro_stream, return_output_stream=True)
            if ica_stream == 2 and np.isfinite(ica_output).all():
                ica_latents.append(ica_output)

            # bw step
            pre_bw_output, pre_bw_stream = {
                'prosvd': (pro_output, pro_stream),
                'jpca': (jpca_output, jpca_stream),
                'mmica': (ica_output, ica_stream),
            }[latents_for_bw]

            if pre_bw_stream == 2 and pre_bw_latent_dims_to_drop > 0:
                pre_bw_output = pre_bw_output[:, :-pre_bw_latent_dims_to_drop]

            output, stream = bw.partial_fit_transform(pre_bw_output, pre_bw_stream, return_output_stream=True)

            # fit all the regressions on alpha
            alpha_to_beh_reg.partial_fit_transform(output, stream)
            neural_stream_output, neural_stream_stream = neural_only_reg_pipeline.partial_fit_transform(output, stream,
                                                                                                        return_output_stream=True)

            alpha_to_joint_latents_reg.partial_fit_transform(output, stream)
            alpha_to_joint_latents_reg.partial_fit_transform(pre_bw_output, 5 if pre_bw_stream == 2 else pre_bw_stream)

            if stream == 3:
                beh_target.append(output)

            if neural_stream_stream == 4 and not np.isnan(neural_stream_output).any():
                neural_target.append(neural_stream_output)

            if pre_bw_stream == 2 and np.isfinite(pre_bw_output).all():
                joint_target.append(pre_bw_output)

            if stream == 2 and np.isfinite(output).all():  # do predictions
                prediction_t = output.t + bw.dt

                if alpha_pred_method == 'normal':
                    alpha_pred = bw.get_alpha_at_t(prediction_t)

                elif alpha_pred_method == 'current_alpha':
                    alpha_pred = bw.get_alpha_at_t(0, relative_t=True)

                elif alpha_pred_method == 'force_one_bubble':
                    alpha_pred = bw.get_alpha_at_t(prediction_t)
                    alpha_pred[np.argmax(bw.alpha)] = 0
                    alpha_pred[alpha_pred < alpha_pred.max()] = 0
                    alpha_pred = alpha_pred / alpha_pred.sum()



                next_bubble_predictions.append(ArrayWithTime(bw.mu[np.argmax(alpha_pred)], t=prediction_t))

                beh_predictions.append(ArrayWithTime(alpha_to_beh_reg.predict(alpha_pred), t=prediction_t))
                neural_predictions.append(ArrayWithTime(alpha_to_neural_reg.predict(alpha_pred), t=prediction_t))
                joint_predictions.append(ArrayWithTime(alpha_to_joint_latents_reg.predict(alpha_pred), t=prediction_t))

            if output.t >= exit_time:
                end_time = timeit.default_timer()
                break

            pbar.update(round(output.t, 1) - pbar.n)

        # these were only used in making videos
        next_bubble_predictions = ArrayWithTime.from_list(next_bubble_predictions)

        self.d = d

        self.pro = pro
        self.ica = ica
        self.jpca = jpca

        self.bw = Bubblewrap(**(bw.get_params()))  # this is supposed to free up jax memory
        self.bw.log = bw.log
        self.bw.dt = bw.dt

        self.pro_latents = ArrayWithTime.from_list(pro_latents)
        self.jpca_latents = ArrayWithTime.from_list(jpca_latents)
        self.ica_latents = ArrayWithTime.from_list(ica_latents)

        self.beh_predictions = ArrayWithTime.from_list(beh_predictions)
        self.neural_predictions = ArrayWithTime.from_list(neural_predictions)
        self.joint_predictions = ArrayWithTime.from_list(joint_predictions)

        self.beh_target = ArrayWithTime.from_list(beh_target)
        self.neural_target = ArrayWithTime.from_list(neural_target)
        self.joint_target = ArrayWithTime.from_list(joint_target)

        self.beh_correlations, self.beh_nrmses = self.evaluate_regression(self.beh_predictions, self.beh_target)[-2:]
        self.neural_correlations, self.neural_nrmses = self.evaluate_regression(self.neural_predictions, self.neural_target)[-2:]
        self.joint_correlations, self.joint_nrmses = self.evaluate_regression(self.joint_predictions, self.joint_target)[-2:]

        self.time_elapsed = end_time - start_time


    @staticmethod
    def evaluate_regression(estimates, targets):
        t = estimates.t
        targets = resample_matched_timeseries(
            targets,
            targets.t,
            estimates.t
        )

        test_s = t > (t[0] + t[-1]) / 2

        correlations = np.array([np.corrcoef(estimates[test_s, i], targets[test_s, i])[0, 1] for i in range(estimates.shape[1])])
        nrmse_s = np.sqrt(((estimates[test_s] - targets[test_s])**2).mean(axis=0))/targets[test_s].std(axis=0)

        return targets, estimates, correlations, nrmse_s


    def plot_results(self):
        behavior_dicts = [
            dict(
                true=self.beh_target,
                predicted=self.beh_predictions,
                label='beh'
            ),
            dict(
                true=self.neural_target,
                predicted=self.neural_predictions,
                label='neural'
            ),
            dict(
                true=self.joint_target,
                predicted=self.joint_predictions,
                label='joint'
            ),
        ]
        self.plot_bw_pipeline([self.bw], behavior_dicts)

    def plot_bw_pipeline(self, bws, behavior_dict_list=None, t_in_samples=False):
        import matplotlib.pyplot as plt
        def _one_sided_ewma(data, com=100):
            import pandas as pd
            return pd.DataFrame(data=dict(data=data)).ewm(com).mean()["data"]

        def plot_with_trendline(ax, times, data, color, com=100):
            ax.plot(times, data, alpha=.25, color=color)
            smoothed_data = _one_sided_ewma(data, com, )
            ax.plot(times, smoothed_data, color=color)

        bws: [adaptive_latents.Bubblewrap]
        assert len(bws) == 1
        for bw in bws:
            assert bw.log_level > 0


        fig, axs = plt.subplots(figsize=(14, 5 + 2 * len(behavior_dict_list)), nrows=2 + len(behavior_dict_list), ncols=2, sharex='col', layout='tight',
                                gridspec_kw={'width_ratios': [7, 1]})

        common_time_start = max([min(bw.log['t']) for bw in bws])
        common_time_end = min([max(bw.log['t']) for bw in bws])
        halfway_time = (common_time_start + common_time_end) / 2

        to_write = [[] for _ in range(axs.shape[0])]
        colors = ['C0'] + ['k'] * (len(bws) - 1)

        for idx, bw in enumerate(bws): # only happens once
            color = colors[idx]

            # plot prediction
            t = np.array(bw.log['log_pred_p_origin_t'])
            t_to_plot = t
            if t_in_samples:
                t_to_plot = t / bw.dt
            to_plot = np.array(bw.log['log_pred_p'])
            plot_with_trendline(axs[0, 0], t_to_plot, to_plot, color)
            last_half_mean = to_plot[(halfway_time < t) & (t < common_time_end)].mean()
            to_write[0].append((idx, f'{last_half_mean:.2f}', {'color': color}))
            axs[0, 0].set_ylabel('log pred. p')

            # plot entropy
            t = np.array(bw.log['t'])
            t_to_plot = t
            if t_in_samples:
                t_to_plot = t / bw.dt
            to_plot = np.array(bw.log['entropy'])
            plot_with_trendline(axs[1, 0], t_to_plot, to_plot, color)
            last_half_mean = to_plot[(halfway_time < t) & (t < common_time_end)].mean()
            to_write[1].append((idx, f'{last_half_mean:.2f}', {'color': color}))
            axs[1, 0].set_ylabel('entropy')

            max_entropy = np.log2(bw.N)
            axs[1, 0].axhline(max_entropy, color='k', linestyle='--')

        # plot behavior
        for idx, behavior_dict in enumerate(behavior_dict_list):
            ax_n = 2 + idx

            targets, estimates, correlations, _ = self.evaluate_regression(behavior_dict['predicted'], behavior_dict['true'])

            corr_str = '\n'.join([f'{r:.2f}' for r in correlations] )
            to_write[ax_n].append((idx, corr_str, {'fontsize': 'large'}))

            t_to_plot = estimates.t
            if t_in_samples:
                t_to_plot = estimates.t / np.median(np.diff(estimates.t))
            for i in range(estimates.shape[1]):
                axs[ax_n,0].plot(t_to_plot, targets[:, i], color=f'C{i}')
                axs[ax_n,0].plot(t_to_plot, estimates[:, i], color=f'C{i}', alpha=.5)
            axs[ax_n,0].set_xlabel("time")
            axs[ax_n,0].set_ylabel(behavior_dict['label'])


        # this sets the axis bounds for the text
        for axis in axs[:, 0]:
            data_lim = np.array(axis.dataLim).T.flatten()
            bounds = data_lim
            bounds[:2] = (bounds[:2] - bounds[:2].mean()) * np.array([1.02, 1.2]) + bounds[:2].mean()
            bounds[2:] = (bounds[2:] - bounds[2:].mean()) * np.array([1.05, 1.05]) + bounds[2:].mean()
            axis.axis(bounds)
            axis.format_coord = lambda x, y: 'x={:g}, y={:g}'.format(x, y)

        # this prints the last-half means
        for i, l in enumerate(to_write):
            for idx, text, kw in l:
                x, y = .92, .93 - .1 * idx
                x, y = axs[i, 0].transLimits.inverted().transform([x, y])
                axs[i, 0].text(x, y, text, clip_on=True, verticalalignment='top', **kw)

        # this creates the axis for the parameters
        gs = axs[0, 1].get_gridspec()
        for a in axs[:, 1]:
            a.remove()
        axbig = fig.add_subplot(gs[:, 1])
        axbig.axis("off")

        # this generates and prints the parameters
        params_per_bw_list = [bw.get_params() for bw in bws]
        super_param_dict = {}
        for key in params_per_bw_list[0].keys():
            values = [p[key] for p in params_per_bw_list]
            if len(set(values)) == 1:
                values = values[0]
                if key in {'input_streams', 'output_streams', 'log_level'}:
                    continue
            super_param_dict[key] = values
        to_write = "\n".join(f"{k}: {v}" for k, v in super_param_dict.items())
        axbig.text(0, 1, to_write, transform=axbig.transAxes, verticalalignment="top")
