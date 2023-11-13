import matplotlib.pyplot as plt
import bubblewrap.plotting_functions as bpf

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from bubblewrap.bw_run import BWRun



def test_axis_plots(make_br):

    fig, axs = plt.subplots(nrows=1, ncols=2)
    ax = axs[0]
    br:BWRun = make_br()
    bw = br.bw
    obs, beh = br.obs_ds.get_history(), br.beh_ds.get_history()

    offset = 3
    predicted_location = br.obs_ds.get_atemporal_data_point(offset=offset)

    ### good ###
    bpf.show_bubbles_2d(ax, obs, bw)
    bpf.show_active_bubbles_2d(ax, obs, bw)
    bpf.show_A(ax, bw)
    bpf.show_alpha(ax, br)
    bpf.show_behavior_variables(ax, br, beh)
    bpf.show_A_eigenspectrum(ax, bw)
    bpf.show_data_distance(ax, obs, max_step=50)

    bpf.show_bubbles_2d(axs[0], obs, bw)
    bpf.show_nstep_pred_pdf(axs[1], br, axs[0], fig=fig, offset=offset)


    ### to fix ###
    # bpf.br_plot_3d(br)
    # bpf.show_Ct_y(ax, regressor=bw.regressor)
    # bpf.show_w_sideways(ax, bw, current_behavior)

def test_comparison_plots(make_br):
    brs = [make_br() for _ in range(3)]
    bpf.compare_metrics(brs, offset=0)