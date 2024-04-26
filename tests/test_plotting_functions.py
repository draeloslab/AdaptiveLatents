import matplotlib.pyplot as plt
import adaptive_latents.plotting_functions as bpf
import adaptive_latents as al

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from adaptive_latents.bw_run import BWRun



def test_axis_plots(premade_unfrozen_br, premade_big_br):

    fig, axs = plt.subplots(nrows=1, ncols=2)
    ax = axs[0]
    br:BWRun = premade_unfrozen_br
    bw = br.bw
    obs, obs_t = br.input_ds.get_history()
    beh, beh_t = br.output_ds.get_history()

    big_br:BWRun = premade_big_br

    neural_offset = br.input_ds.time_offsets[-1] # this is usually 1
    behavior_offset = br.output_ds.time_offsets[-1] # this is usually 1

    bpf.show_bubbles_2d(ax, obs, bw)
    bpf.show_data_2d(ax, obs, bw)
    bpf.show_active_bubbles_2d(ax, obs, bw)
    bpf.show_A(ax, fig, bw)
    bpf.show_A(ax, fig, bw, show_log=True)
    bpf.show_B(ax, br)
    bpf.show_B(ax, br, show_log=True)
    bpf.show_alpha(ax, br)
    bpf.show_alpha(ax, big_br, show_log=True)
    bpf.show_behavior(ax, br, offset=behavior_offset)
    bpf.show_A_eigenspectrum(ax, bw)
    bpf.show_data_distance(ax, obs, max_step=50)

    bpf.show_active_bubbles_and_connections_2d(ax, obs, bw)
    bpf.show_nstep_pdf(axs[1], br, axs[0], fig=fig, offset=neural_offset)

    hmm = al.input_sources.hmm_simulation.HMM.gaussian_clock_hmm(n_states=12) # this is hacky, you should use the same HMM as generated the data
    bpf.show_nstep_pdf(axs[1], br, axs[0], fig=fig, offset=neural_offset, method="hmm", hmm=hmm)

def test_comparison_plots(make_br):
    brs = [make_br(bw_params=dict(B_thresh=-15+i)) for i in range(3)]
    bpf.compare_metrics(brs, offset=0, include_behavior=True)
    bpf.compare_metrics(brs, offset=3, minutes=True, red_lines=(50,))

# TODO:
#     test_redlines_on_attribute_of_bwrun