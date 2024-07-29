import matplotlib.pyplot as plt
import adaptive_latents.plotting_functions as bpf
import adaptive_latents as al

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from adaptive_latents.bw_run import BWRun


# def test_axis_plots(premade_unfrozen_br, show_plots):
#
#     fig, axs = plt.subplots(nrows=5, ncols=5, layout='tight', figsize=(10, 10))
#     axes = (a for a in axs.flatten())
#     br: BWRun = premade_unfrozen_br
#     bw = br.bw
#     obs, obs_t = br.input_ds.get_history()
#     beh, beh_t = br.output_ds.get_history()
#
#     neural_offset = br.input_ds.time_offsets[-1]  # this is usually 1
#     behavior_offset = br.output_ds.time_offsets[-1]  # this is usually 1
#
#     bpf.show_bubbles_2d(next(axes), obs, bw)
#     bpf.show_data_2d(next(axes), obs, bw)
#     bpf.show_active_bubbles_2d(next(axes), obs, bw)
#     bpf.show_A(next(axes), fig, bw)
#     bpf.show_A(next(axes), fig, bw, show_log=True)
#     bpf.show_B(next(axes), br)
#     bpf.show_B(next(axes), br, show_log=True)
#     bpf.show_alpha(next(axes), br)
#     bpf.show_behavior(next(axes), br, offset=behavior_offset)
#     bpf.show_A_eigenspectrum(next(axes), bw)
#     bpf.show_data_distance(next(axes), obs, max_step=50)
#
#     # big_br: BWRun = premade_big_br
#     # bpf.show_alpha(ax, big_br, show_log=True)
#
#     bpf.show_active_bubbles_and_connections_2d(next(axes), obs, bw)
#
#     next(axes).axis("off")
#     ax1, ax2 = next(axes), next(axes)
#     bpf.show_nstep_pdf(ax2, br, ax1, fig=fig, offset=neural_offset)
#
#     next(axes).axis("off")
#     ax1, ax2 = next(axes), next(axes)
#     hmm = al.input_sources.hmm_simulation.HMM.gaussian_clock_hmm(n_states=12)  # this is hacky, you should use the same HMM as generated the data
#     bpf.show_nstep_pdf(ax2, br, ax1, fig=fig, offset=neural_offset, method="hmm", hmm=hmm)
#
#     if show_plots:
#         for ax in axes:
#             ax.axis('off')
#         plt.show()
#
#
# def test_comparison_plots(make_br):
#     brs = [make_br(bw_params=dict(B_thresh=-15 + i, M=100 + 10*i)) for i in range(3)]
#     bpf.compare_metrics(brs, offset=0, include_behavior=True)
#     bpf.compare_metrics(brs, offset=3, minutes=True, red_lines=(50,))


# TODO:
#     test_redlines_on_attribute_of_bwrun
