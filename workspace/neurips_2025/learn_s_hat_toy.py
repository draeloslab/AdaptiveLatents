import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive_latents.input_sources.lds_simulation import LDS
from adaptive_latents import StreamingKalmanFilter, ArrayWithTime, Pipeline, StimRegressor, Bubblewrap
import tqdm.auto as tqdm

from common import log_for_tex
from make_constants_file import to_tex_command

standard_kinds_of_sr = ['learning from stim', 'ignoring stim samples', 'unaware of stim']
time_slices = ('post-stim', 'non-stim', 'all')
space_slices = ('stim-d', 'non-stim-d', 'all')

n_rotations = 50
noise_variance = 0.05
stims_per_rotation = 2
stim_magnitude = 10


def make_slices_tensor(sr):
    error = sr.log['pred_error']
    stim_intended_samples = sr.log['stim_intended_samples']

    outputs = []
    index_one = []
    index_two = []

    for i, time_slice in enumerate(time_slices):
        match time_slice:
            case 'post-stim':
                _, value = ArrayWithTime.align_indices(stim_intended_samples, error)
            case 'non-stim':
                _, value = ArrayWithTime.align_indices(stim_intended_samples, error, complement=True)
            case 'all':
                value = error
            case _:
                raise ValueError()

        for j, space_slice in enumerate(space_slices):
            match space_slice:
                case 'stim-d':
                    value2 = value[:,2:3]
                case 'non-stim-d':
                    value2 = value[:,:2]
                case 'all':
                    value2 = value
                case _:
                    raise ValueError()

            outputs.append(value2)
            index_one.append(time_slice)
            index_two.append(space_slice)
    outputs = pd.Series(outputs, index=[np.array(index_one), np.array(index_two)])
    return outputs

from learn_s_hat_plots import finalize_log

def make_ideal_nostim_srs(rng, n_runs=1, streaming=False, show_tqdm=False):
    ideal_srs = []
    for _ in tqdm.trange(n_runs, disable=not show_tqdm):
        if not streaming:
            X, Y, stim = LDS.run_nest_dynamical_system(n_rotations, stims_per_rotation=stims_per_rotation, stim_magnitude=0, rng=rng, u_function='curvy', noise=noise_variance)
            kf = StreamingKalmanFilter(steps_between_refits=float('inf'))
            kf.fit(Y, Y)  # Y, Y?


            sr_ideal = StimRegressor(autoreg=kf, log_level=2, check_dt=True)
            sr_ideal.offline_run_on([(Y, 'X'), (stim, 'stim')], convinient_return=False, show_tqdm=False)
            ideal_srs.append(finalize_log(sr_ideal, stim))
        else:
            sr_ideal = StimRegressor(autoreg=StreamingKalmanFilter(), log_level=2, check_dt=True)
            _, Y, stim = LDS.run_nest_dynamical_system(n_rotations, stims_per_rotation=stims_per_rotation, stim_magnitude=0, rng=rng, u_function='curvy', noise=noise_variance)
            sr_ideal.offline_run_on([(Y, 'X'), (stim, 'stim')], convinient_return=False, show_tqdm=False)
            ideal_srs.append(finalize_log(sr_ideal, stim))


    return ideal_srs


def true_S(point):
    true = np.zeros(3)
    true[2] = stim_magnitude * point[0] / np.linalg.norm(point[:2])
    return true

def make_s_hat_error_function(rng, n_runs=10, n_points=200):
    previous_Ys = []
    for _ in range(n_runs):
        _, Y, stim = LDS.run_nest_dynamical_system(n_rotations, stims_per_rotation=stims_per_rotation, stim_magnitude=stim_magnitude, rng=rng, u_function='curvy', noise=noise_variance)
        previous_Ys.append(Y)


    test_points = rng.choice(np.vstack(previous_Ys), replace=False, size=n_points)
    def s_hat_error_function(self:StimRegressor):
        s_hat_errors = []
        for point in test_points:
            e = self.stim_reg.predict(np.hstack([point, np.array([1])])) - true_S(point)
            s_hat_errors.append(e)
        return np.linalg.norm(s_hat_errors, axis=1).mean()

    return s_hat_error_function

def single_make_srs(rng):
    _, Y, stim = LDS.run_nest_dynamical_system(n_rotations, stims_per_rotation=stims_per_rotation, stim_magnitude=stim_magnitude, rng=rng, u_function='curvy', noise=noise_variance)

    sr1 = StimRegressor(autoreg=StreamingKalmanFilter(), log_level=2, check_dt=True)
    sr2 = StimRegressor(autoreg=StreamingKalmanFilter(), log_level=2, check_dt=True, attempt_correction=False)
    sr3 = StimRegressor(autoreg=StreamingKalmanFilter(), log_level=2, check_dt=True, attempt_correction=False, heed_stimuli=False)

    sr3.stim_reg.observe(np.zeros(4), np.zeros(3))  # setting a zero prior for the manifold comparison
    s_hat_error_function = make_s_hat_error_function(rng)
    sr1.s_hat_error_function = s_hat_error_function
    # sr2.s_hat_error_function = s_hat_error_function
    sr3.s_hat_error_function = s_hat_error_function

    pre_srs = {'learning from stim': sr1, 'ignoring stim samples':sr2, 'unaware of stim':sr3}
    for sr in pre_srs.values():
        sr.offline_run_on([(Y, 'X'), (stim, 'stim')], convinient_return=False, show_tqdm=False)
        finalize_log(sr, stim)

    return pre_srs

def make_srs(rng, n_runs=1, show_tqdm=False):
    srs = []
    for _ in tqdm.trange(n_runs, disable=not show_tqdm):
        srs.append(single_make_srs(rng))
        assert set(srs[-1].keys()) == set(standard_kinds_of_sr)

    srs = {k:[run[k] for run in srs] for k in srs[0].keys()}
    return srs


from learn_s_hat_plots import plot_onestep_pred_error_decreasing, make_table, plot_manifold_error


if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    parser.add_argument( "--type-of-plot", type=str, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(1)

    fig = None
    match args.type_of_plot:
        case '1-step-prediction':
            srs = make_srs(rng, n_runs=1, show_tqdm=True)

            standard_kinds_of_sr = ['learning from stim', 'ignoring stim samples', 'unaware of stim']
            row_info = [
                dict(time_slice_type='all', space_slice_type='stim-d', title='all-time stim-dimension prediction errors', sr_kind_keys=standard_kinds_of_sr),
                dict(time_slice_type='post-stim', space_slice_type='stim-d', title='post-stim stim-dimension prediction errors', sr_kind_keys=standard_kinds_of_sr, last_half_average=True)
            ]
            fig = plot_onestep_pred_error_decreasing(srs, row_info, make_slices_tensor)

            for lines in fig.axes[1].get_lines():
                ydata = lines.get_ydata()
                if len(ydata) == 2:
                    fig.axes[0].axhline(ydata[0], color=lines.get_color(), linestyle='--')

            h = srs['learning from stim'][0].stim_reg.history
            h = h[~np.isnan(h).any(axis=1)]

            def true_S(state):
                u = np.zeros(3)
                u[2] = stim_magnitude * state[0] / np.linalg.norm(state[:2])
                return u


            fig2, ax2 = plt.subplots(subplot_kw={'projection': '3d', 'computed_zorder':False}, layout='constrained')

            extent = 20
            depth = 14
            X, Y = np.meshgrid(np.linspace(-extent, extent, depth), np.linspace(-extent, extent, depth))
            Z = 0 * X
            for i_x, i_y in itertools.product(range(depth), range(depth)):
                Z[i_x, i_y] = true_S([X[i_x, i_y], Y[i_x, i_y], None])[2]
            ax2.plot_surface(X, Y, Z, color='gray')

            errors = [np.linalg.norm(true_S(row[:3]) - row[-3:]) for row in h]
            c = ax2.scatter(h[:,0], h[:,1], h[:,-1], c=errors, zorder=100, alpha=1, cmap='plasma', vmin=0)
            fig2.colorbar(c)

            ax2.axis('equal')
            ax2.view_init(elev=24, azim=147, roll=0)
            ax2.axis((np.float64(-24.059680968092277), np.float64(27.5328712068437), np.float64(-26.059286658753816), np.float64(27.257531995356583), np.float64(-12.057545210133634), np.float64(11.79530183372015)))

            ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax2.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax2.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax2.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

            fig2.savefig(args.output.with_stem('toy_curvy'), bbox_inches="tight")


        case '1-step-prediction-table':
            n_runs = 5
            srs = make_srs(rng, n_runs=n_runs, show_tqdm=True)
            srs['ideal'] = make_ideal_nostim_srs(rng, n_runs=n_runs, streaming=False, show_tqdm=True)
            srs['ideal streaming'] = make_ideal_nostim_srs(rng, n_runs=n_runs, streaming=True, show_tqdm=True)
            table_text, _, _ = make_table(srs, time_slices=['post-stim', 'non-stim'], space_slices=['stim-d', 'non-stim-d'], make_slices_tensor=make_slices_tensor, show_rows=False, normalize_key='ideal')

            with open(args.output,'w') as fhan:
                fhan.write(to_tex_command(key='s_hat_toy_rmse_comparison_table', value=table_text))

        case 'manifold-error':
            srs = make_srs(rng, n_runs=1, show_tqdm=False)
            fig = plot_manifold_error(srs)
        case _:
            raise ValueError()


    if fig is not None:
        fig.savefig(args.output, bbox_inches="tight")
