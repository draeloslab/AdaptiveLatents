import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def make_native_nearness_to_offline_figure(offline_errors, trajectories, t, i, name, offline_error_t=None):
    colors = ["#F94B00", "#2FA194", "#EC3C8E"]
    common_ylim = [.005, 4]
    common_xticks = [0, 30, 60]


    fig, ax = plt.subplots()
    fig: plt.Figure

    if not hasattr(offline_errors[0], "__len__"):
        lower = np.quantile(offline_errors, stats.norm(0, 1).cdf(-1))
        upper = np.quantile(offline_errors, stats.norm(0, 1).cdf(1))
        ax.axhline(np.quantile(offline_errors, .5), color='k', alpha=.75)
        ax.fill_between([t[0], t[-1]], upper, lower, color='k', edgecolor=None, alpha=.25)
    else:
        lower = np.nanquantile(offline_errors, stats.norm(0, 1).cdf(-1), axis=0)
        upper = np.nanquantile(offline_errors, stats.norm(0, 1).cdf(1), axis=0)
        trajectory = np.nanmedian(offline_errors, axis=0)
        ax.plot(offline_error_t, trajectory, color='k', alpha=.75)
        ax.fill_between(offline_error_t, upper, lower, color='k', edgecolor=None, alpha=.25)

    ax.autoscale(False)

    lower = np.quantile(trajectories, stats.norm(0, 1).cdf(-1), axis=0)
    upper = np.quantile(trajectories, stats.norm(0, 1).cdf(1), axis=0)
    trajectory = np.median(trajectories, axis=0)
    ax.plot(t, trajectory, color=colors[i])
    ax.fill_between(t, upper, lower, color=colors[i], edgecolor=None, alpha=.5)

    ax.set_xlabel('time (s)')
    ax.set_ylabel('error')
    ax.set_xticks(common_xticks)

    ax.semilogy()
    ax.set_yticks([1, .1, .01])
    ax.minorticks_off()
    ax.set_ylim(common_ylim)
    ax.set_xticks(common_xticks)

    ax.set_title(f"{name} error")

    return fig
