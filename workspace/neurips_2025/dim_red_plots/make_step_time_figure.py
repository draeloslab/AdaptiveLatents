import matplotlib.pyplot as plt


def make_step_time_figure(step_times):
    slice_times = step_times.slice_by_time(slice(30, None))
    fig, ax = plt.subplots()
    to_plot = slice_times * 1000
    ax.plot(slice_times.t - step_times.t.min(), to_plot)
    ax.set_ylabel("time per step (ms)")
    ax.set_xlabel("experiment time (s)")
    ax.set_ylim(0.0, 1.5)
    return fig
