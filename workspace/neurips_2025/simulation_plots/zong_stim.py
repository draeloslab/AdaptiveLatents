import numpy as np
from adaptive_latents import datasets
from adaptive_latents.stim_regressor import StimAutoReg
from adaptive_latents.sim_stim import *
import tqdm.autonotebook as tqdm
import copy
import matplotlib.pyplot as plt
from adaptive_latents.utils import save_to_cache

def invert(x):
    return 1 / x

zero_thresh = 0.05
amount_to_add = 4
switch_time = 304
colors = ['#ca1469ff','#4d4d4dff']


#         line.set_color()
#     elif color == 'C1':
#         line.set_color('#4d4d4dff')


def new_make_sr(
        input_array,
        rng,
        autoreg=StreamingKalmanFilter,
        stim_rate=1, # TODO: refactor out
        regular_stim_iter=None,  # TODO: refactor out
        isi_generator=None,
        exit_time=60,
        decay_rate=.8,
        prosvd_k=10,
        stim_magnitude=10,
        max_l0_norm=30,
        attempt_correction=True,
        heed_stimuli=True,
        stim_time_delay=0,
        regressor_stim_delay=0,
        design_method='optimized identity u_to_s', # TODO: refactor out
        design_type=None,
        u_to_s_model_type=None,
        true_S='identity',
        stim_timing_method='random',
        n_identity_prior=10,
        stim_direction_type='first',
        initial_nostim_period=5,
        stim_reg_maxlen=500,
        smoothing_tau=None,
        centerer_init_size=0,
        last_dim_red='prosvd',
        show_tqdm=False,
):
    _init_time = time.time()
    timing_log = SimpleNamespace()
    timing_log.init_time = _init_time
    timing_log.loop_time = 0
    timing_log.stim_design = []
    timing_log.dimension_reduction = []
    timing_log.sr_update = []
    timing_log.per_loop = []
    timing_log.stim_reg_updated = []
    timing_log.in_sim_time = []



    assert (regular_stim_iter is not None) + (stim_rate is not None) + (isi_generator is not None) == 1
    if stim_rate:
        isi_generator = cycle([1/stim_rate])
    elif regular_stim_iter:
        isi_generator = map(invert, regular_stim_iter)
        assert stim_timing_method == 'regular'
        stim_timing_method = 'isi'
    del regular_stim_iter, stim_rate

    optimization_method, u_to_s_model_type = {
        'optimized learned u_to_s': ('jaxopt', 'kernel_regressed'),
        'optimized identity u_to_s': ('jaxopt', 'identity'),
        'direct cheating': ('cheat_lowd_vec', 'identity'),
        'single neurons': ('cheat_highd_vec_single_neurons', None),
        'many neurons': ('cheat_highd_vec_many_neurons', None),
    }[design_method]
    # single neurons
    # many neurons
    del design_method

    stim_time_rng, other_rng = rng.spawn(2)


    sr = StimRegressor(
        autoreg=autoreg(),
        stim_reg=BaseMultiKernelRegressor(length_scales=[0.04, 0.04, 0.04], maxlen=stim_reg_maxlen),
        log_level=2,
        check_dt=True,
        attempt_correction=attempt_correction,
        heed_stimuli=heed_stimuli,
        stim_delay=regressor_stim_delay,
    )
    sr.stim_autoreg = StimAutoReg(n_steps_to_consider=6)

    stim_designer = StimDesigner(
        max_l0_norm=max_l0_norm,
        rng_seed=other_rng.integers(2 ** 32),
        should_log=True,
        initial_nostim_period=initial_nostim_period,
        stim_timing_method=stim_timing_method,
        inter_stim_interval_generator=isi_generator,
        optimization_method=optimization_method, # todo:fix
        u_to_s_model_type=u_to_s_model_type,
        n_identity_initialization=n_identity_prior
    )

    static_S_seed = other_rng.integers(2 ** 32)
    sim_stim_adder = SimulatedStimAdder(
        true_S=true_S,
        static_S_seed=static_S_seed,
        stim_time_delay=stim_time_delay,
        decay=decay_rate
    )
    ss_adder_p = Pipeline([
        sim_stim_adder,
        KernelSmoother(tau=1)
    ])

    log = {}

    centerer = CenteringTransformer(init_size=centerer_init_size, nan_when_uninitialized=True)
    if smoothing_tau is not None:
        smoother = KernelSmoother(tau=smoothing_tau/input_array.dt)
    else:
        smoother = Pipeline()

    pro = proSVD(k=prosvd_k)
    if last_dim_red == 'prosvd':
        last_dim_red_object = None
    elif last_dim_red == 'sjpca':
        last_dim_red_object = sjPCA()
    elif last_dim_red == 'mmica':
        last_dim_red_object = mmICA()
    else:
        raise ValueError()

    decided_stims = []
    latents = []
    high_d_without_stim = []
    high_d_with_stim = []
    high_d_stims = []

    pbar = nullcontext()
    if show_tqdm:
        pbar = tqdm(total=min(input_array.t[-1], exit_time))

    timing_log.init_time = time.time() - timing_log.init_time
    timing_log.loop_time = time.time()
    has_changed = False
    with pbar:
        for data in Pipeline().streaming_run_on(input_array):

            if data.t > switch_time and not has_changed:
                print(f'delay is now {amount_to_add * input_array.dt}')
                sim_stim_adder.stim_delay_queue = deque([0] * amount_to_add)
                sr.stim_delay = sr.stim_delay + sr.dt * amount_to_add
                has_changed = True

            timing_log.in_sim_time.append(data.t)
            timing_log.per_loop.append(time.time())
            timing_log.stim_design.append(time.time())

            stim_decision = stim_designer.decide_whether_to_stim(data.t, stim_time_rng=stim_time_rng, input_array_dt=input_array.dt)
            decided_stims.append(ArrayWithTime(stim_decision, data.t))

            equivalent_projection_matrix = calculate_equivalent_projection_matrix(pro, last_dim_red_object)
            if stim_decision and equivalent_projection_matrix is not None:
                desired_stim = stim_designer.desired_stim_direction(equivalent_projection_matrix, stim_direction_type, other_rng)
                designed_stim = stim_designer.sim_stim_design_stim(sr, stim_magnitude, desired_stim, equivalent_projection_matrix, current_t=data.t)
                instantaneous_stim = designed_stim * stim_magnitude
                instantaneous_stim[np.abs(instantaneous_stim) < zero_thresh] = 0
            else:
                instantaneous_stim = np.zeros(input_array.shape[1])
            timing_log.stim_design[-1] = time.time() - timing_log.stim_design[-1]

            true_stim_result = sim_stim_adder.true_stim_result(instantaneous_stim, equivalent_projection_matrix)

            ss_adder_p.partial_fit_transform(true_stim_result, stream='stim')

            high_d_without_stim.append(data)
            pre_stim_data = data
            data = ss_adder_p.partial_fit_transform(data, stream='X')
            high_d_with_stim.append(data)
            high_d_stims.append(data - pre_stim_data)

            timing_log.dimension_reduction.append(time.time())
            data = centerer.partial_fit_transform(data, stream= 'X')
            data = smoother.partial_fit_transform(data, stream= 'X')
            data = pro.partial_fit_transform(data, stream='X')
            if last_dim_red_object is not None:
                data = last_dim_red_object.partial_fit_transform(data, stream='X')
            timing_log.dimension_reduction[-1] = time.time() - timing_log.dimension_reduction[-1]
            latents.append(data)

            timing_log.stim_reg_updated.append(sr.stim_reg.n_observed)
            timing_log.sr_update.append(time.time())
            sr.partial_fit_transform(ArrayWithTime(true_stim_result, data.t), stream= 'stim')
            stims_before_obs = set([stim.t for stim in sr.last_seen_stims])
            data = sr.partial_fit_transform(data, stream= 'X')
            resolved_stim_ts = stims_before_obs - set([stim.t for stim in sr.last_seen_stims])
            timing_log.sr_update[-1] = time.time() - timing_log.sr_update[-1]
            timing_log.stim_reg_updated[-1] = timing_log.stim_reg_updated[-1] != sr.stim_reg.n_observed

            if heed_stimuli and len(resolved_stim_ts):
                assert len(resolved_stim_ts) == 1
                stim_t = list(resolved_stim_ts)[0]
                for l in reversed(stim_designer.log):
                    if stim_t == l['time_of_stim']:
                        obs = sr.stim_reg.get_obs(t=stim_t + sr.stim_delay)
                        l['observed_s_hat'] = obs.pop('output')
                        l['observed_reg_input'] = [v for v in obs.values()]
                        break
                else:
                    raise Exception('resolved stim is not in stim_designer log')

            if show_tqdm:
                pbar.update(round(float(data.t), 2) - pbar.n)

            timing_log.per_loop[-1] = time.time() - timing_log.per_loop[-1]
            if data.t > exit_time:
                break

    timing_log.loop_time = time.time() - timing_log.loop_time
    log['high_d_stims'] = ArrayWithTime.from_list(high_d_stims, squeeze_type='to_2d', drop_early_nans=False)
    log['high_d_without_stim'] = ArrayWithTime.from_list(high_d_without_stim, squeeze_type='to_2d', drop_early_nans=False)
    log['high_d_with_stim'] = ArrayWithTime.from_list(high_d_with_stim, squeeze_type='to_2d', drop_early_nans=False)
    assert np.allclose(log['high_d_with_stim'], log['high_d_stims'] + log['high_d_without_stim'], equal_nan=True)
    log['latents'] = ArrayWithTime.from_list(latents, squeeze_type='to_2d', drop_early_nans=True)
    if (log['high_d_stims'] == 0).all():
        warnings.warn("No stims delivered in sim-stim.")

    stim_intended_samples = ArrayWithTime.from_list(decided_stims, squeeze_type='to_2d')
    log['stim_intended_samples'] = stim_intended_samples.slice((stim_intended_samples > 0).any(axis=1))
    log['timing_log'] = timing_log

    sr.log['pred_error'] = ArrayWithTime.from_list(sr.log['pred_error'])



    return sr, stim_designer, log



def make_sr(*args, **kwargs):
    sr, stim_designer, log = new_make_sr(*args, **kwargs)
    sr.log.update(log)
    sr.stim_designer = stim_designer

    # sr.log['stim_intended_samples'] = ArrayWithTime.from_list(sr.log['stim_intended_samples'])
    return sr

def make_srs(data, rng, comparison_preset=None, n_runs=1, show_tqdm=False, overrides=None):
    if overrides is None:
        overrides = {}

    common = dict(stim_magnitude = 10, design_method = 'optimized identity u_to_s', exit_time = np.inf, stim_rate = None, smoothing_tau = 1, centerer_init_size = 8 * 25, initial_nostim_period = 30, regular_stim_iter = cycle([1 / 10, 1 / 3]), stim_timing_method = 'regular', autoreg=functools.partial(StreamingKalmanFilter, steps_between_refits=5))

    to_run = {
        'learning from stim': common | dict(attempt_correction=True, heed_stimuli=True),
        # 'ignoring stim': common | dict(attempt_correction=False, heed_stimuli=True),
        'unaware of stim': common | dict(attempt_correction=False, heed_stimuli=False),
    }

    srs = {}
    with tqdm.tqdm(total=len(to_run) * n_runs, disable=not show_tqdm) as pbar:
        for key, val in to_run.items():
            val = val | overrides
            sub_rng = copy.deepcopy(rng)
            srs[key] = []
            for _ in range(n_runs):
                srs[key].append(make_sr(input_array=data, rng=sub_rng, **val))
                pbar.update(1)

    return srs

def plot_1(srs):
    i = 40
    sr = srs['learning from stim'][0]

    fig_1, axs = plt.subplots(ncols=2, figsize=(10,4), sharex=False, sharey=False, layout='constrained')

    latents = sr.log['latents'].slice_by_time(slice(30,None))
    axs[0].plot(latents[:, 0], latents[:, 1], alpha=.1, color='k')
    stim_s = sr.log['stim_intended_samples'].t - latents.dt

    l = 1
    r = 4.7
    ax_n = 0
    center_t = sr.log['stim_intended_samples'].t[i]
    latents = sr.log['latents'].slice_by_time(slice(center_t-l,center_t+r))
    line = axs[ax_n].plot(latents[:, 0], latents[:, 1], color='k', lw=3)
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
    # n_nonzero = np.linalg.norm(u,ord=0)
    n_nonzero = (np.abs(u) > zero_thresh).sum() # these were actually zeroed out with a custom line, this isn't a threshold
    print(f'{n_nonzero=}')

    high_d = sr.log['high_d_with_stim'].slice_by_time(slice(center_t-l,center_t+r))
    axs[1].plot(high_d.t, high_d[:,idx[:int(n_nonzero)]], color='k', lw=1)
    axs[1].set_xticks([302, 304, 306,308])
    for stim_t in stim_s:
        axs[1].axvline(stim_t, color='r')
    return fig_1



from sim_stim import make_slices_tensor

def plot_onestep_pred_error_decreasing(srs, row_info, make_slices_tensor):
    fig, axs = plt.subplots(nrows=len(row_info), layout='tight', figsize=(8, 2*len(row_info)+1), sharex=True, sharey=True)

    def p(ax, time_slice_type, space_slice_type, xlabel='time', sr_kind_keys=None, title=None, time_slice=None, last_half_average=False):
        if time_slice is  None:
            time_slice = slice(None, None)

        if sr_kind_keys is None:
            sr_kind_keys = srs.keys()
        for idx, sr_kind_key in reversed(list(enumerate(sr_kind_keys))):
            all_to_plot = []
            for sr in srs[sr_kind_key]:
                run_to_plot = make_slices_tensor(sr)
                sub_to_plot = run_to_plot[time_slice_type][space_slice_type].slice_by_time(time_slice)
                sub_to_plot = ArrayWithTime(np.linalg.norm(sub_to_plot, axis=1), sub_to_plot.t)
                all_to_plot.append(sub_to_plot)
            to_plot = ArrayWithTime(np.hstack(all_to_plot), np.hstack([p.t for p in all_to_plot])) # TODO: sort by time
            # TODO: you could do smoothing here
            ax.plot(to_plot.t, to_plot, '.-', color=f'C{idx}', label=sr_kind_key)
            if last_half_average:
                halfway = (to_plot.t.max() + to_plot.t.min()) / 2
                mean = float(to_plot.slice_by_time(slice(halfway, None)).mean())
                ax.axhline(mean, linestyle='--', color=f'C{idx}')
        # ax.legend(loc='upper right')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('error norm')
        if title is None:
            title = f"time:'{time_slice_type}' space:'{space_slice_type}' norm error"
        ax.set_title(title)

    for idx, values in enumerate(row_info):
        p(ax=axs[idx], **values)

    return fig

def plot_2(srs):

    fig, axs = plt.subplots(nrows=1, figsize=np.array((9,6)), squeeze=False, layout='constrained', sharex=False, sharey=True)

    ax = axs[0,0]
    error = srs['unaware of stim'][0].log['pred_error']
    norm_error = np.linalg.norm(error, axis=(1,2))
    ax.plot(error.t, norm_error, '.-', color=colors[1], label='unaware of stim')

    error = srs['learning from stim'][0].log['pred_error']
    norm_error = np.linalg.norm(error, axis=(1,2))
    ax.plot(error.t, norm_error, '.-', color=colors[0], label='learning from stim')

    sr = srs['unaware of stim'][0]
    error = sr.log['pred_error']
    stim_intended_samples = sr.log['stim_intended_samples']
    stim_intended_samples.t[stim_intended_samples.t > switch_time] += amount_to_add * error.dt
    sliced_error, _ = ArrayWithTime.align_indices(error, stim_intended_samples)
    bin_slice = np.array([int(t in sliced_error.t) for t in error.t])
    bin_slice = np.convolve(bin_slice, np.array([0,0,0,0,1,1,1,1,1,1]), mode='same').astype(bool)
    sliced_error = error.slice(bin_slice)
    error_norms = np.linalg.norm(sliced_error, axis=(1,2))
    axs[0,0].axhline(np.nanmean(error_norms), linestyle='--', color=colors[1])
    print(f'unaware mean stim-centered error:  {np.nanmean(error_norms):.3f}')

    sr = srs['learning from stim'][0]
    error = sr.log['pred_error']
    stim_intended_samples = sr.log['stim_intended_samples']
    stim_intended_samples.t[stim_intended_samples.t > switch_time] += amount_to_add * error.dt
    sliced_error, _ = ArrayWithTime.align_indices(error, stim_intended_samples)
    bin_slice = np.array([int(t in sliced_error.t) for t in error.t])
    bin_slice = np.convolve(bin_slice, np.array([0,0,0,0,1,1,1,1,1,1]), mode='same').astype(bool)
    sliced_error = error.slice(bin_slice)
    error_norms = np.linalg.norm(sliced_error, axis=(1,2))
    axs[0,0].axhline(np.nanmean(error_norms), linestyle='--', color=colors[0])

    ax.set_ylim(0, .8)
    ax.axvline(switch_time, linestyle='--', color='gray')

    return fig

def main():
    @save_to_cache('zong_f')
    def f():
        rng = np.random.default_rng(0)
        d = datasets.Zong22Dataset()
        data = d.neural_data

        srs = make_srs(data, rng, comparison_preset='visualization', n_runs=1, show_tqdm=True, overrides=dict(stim_magnitude=9.85, regressor_stim_delay=0*data.dt))
        return srs

    srs = f(_recalculate_cache_value=False)


    fig_1 = plot_1(srs)
    fig_2 = plot_2(srs)

    return fig_1, fig_2


if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    fig_1, fig_2 = main()

    fig_1.savefig(args.output, bbox_inches="tight")
    fig_2.savefig(args.output.with_stem('zong_1step'), bbox_inches="tight")
