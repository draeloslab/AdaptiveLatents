import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from adaptive_latents import ArrayWithTime, datasets
from make_constants_file import to_tex_command
from sim_stim import make_srs, make_slices_tensor

from learn_s_hat_plots import plot_onestep_pred_error_decreasing, make_table


if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    parser.add_argument( "--type-of-plot", type=str, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(1)
    d = datasets.Odoherty21Dataset()
    data = d.neural_data

    fig = None
    match args.type_of_plot:
        case '1-step-prediction':
            srs = make_srs(data, rng, comparison_preset='default', n_runs=1, show_tqdm=False)

            row_info = [
                dict(time_slice_type='all', space_slice_type='stim-d'),
                dict(time_slice_type='post-stim', space_slice_type='stim-d'),
                dict(time_slice_type='all', space_slice_type='non-stim-d'),
                dict(time_slice_type='post-stim', space_slice_type='non-stim-d')
            ]
            fig = plot_onestep_pred_error_decreasing(srs, row_info, make_slices_tensor)

            table_text, _, _ = make_table(srs, time_slices=['post-stim', 'non-stim'], space_slices=['non-stim-d','stim-d'], make_slices_tensor=make_slices_tensor, show_rows=False)
            tex_text = to_tex_command(key='s_hat_ss_rmse_comparison_table', value=table_text)
            (pathlib.Path(args.output).parent / 'learn_s_hat_table_ss.tex').write_text(tex_text)


            with open(pathlib.Path(args.output).parent / 'optimization_history.pkl', 'wb') as fhan:
                # TODO: add this to makefile
                example_sr = srs['learning from stim'][0]
                example_sr.predictions = []
                example_sr.unevaluated_log_pred_ps = []
                pickle.dump(example_sr, fhan)

        case 'delay-table':
            # LDS:
            rng = np.random.default_rng(4)
            from adaptive_latents.input_sources.lds_simulation import LDS
            _, data, _ = LDS.circular_lds(rng=rng).simulate(100, rng=rng)

            srs = make_srs(data, rng, comparison_preset='delay-table', n_runs=5, show_tqdm=True)

            table_text, _, means_table = make_table(srs, time_slices=['all'], space_slices=['non-stim-d','stim-d','all'], make_slices_tensor=make_slices_tensor)
            tex_text = to_tex_command(key='delay-table-ss', value=table_text)
            args.output.write_text(tex_text)

            conditions = np.array([eval(x) for x in means_table.index])
            shape = (len(np.unique(conditions[:,0])), len(np.unique(conditions[:,1])))
            conditions = conditions.reshape(shape + (2,))
            assert (conditions[0,0] == (0,0)).all()
            assert (conditions[1,0] == (1,0)).all()
            assert (conditions[0,1] == (0,1)).all()

            performance_array = means_table['all']['all'].to_numpy().reshape(shape)

            fig, ax = plt.subplots()
            ax.matshow(performance_array)
            ax.set_xlabel('regressor stim delay')
            ax.set_ylabel('real stim delay')
            args.output = args.output.with_suffix('.pdf')

    # case '1-step-prediction-table':
        #     n_runs = 50
        #     srs = make_srs(rng, n_runs=n_runs, show_tqdm=True)
        #     ideal_srs = make_ideal_nostim_srs(rng, n_runs=n_runs, streaming=False, show_tqdm=True)
        #     ideal_streaming_srs = make_ideal_nostim_srs(rng, n_runs=n_runs, streaming=True, show_tqdm=True)
        #     table_text = make_table(srs, ideal_srs, ideal_streaming_srs)
        #
        #     from make_constants_file import to_tex_command
        #     with open(args.output,'w') as fhan:
        #         fhan.write(to_tex_command(key='s_hat_rmse_comparison', value=table_text))

        # case 'manifold-error':
        #     srs = make_srs(rng, n_runs=1, show_tqdm=False)
        #     fig = plot_manifold_error(srs)
        case _:
            raise ValueError()


    if fig is not None:
        fig.savefig(args.output, bbox_inches="tight")
