import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
from math import atan2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bubblewrap
    from bubblewrap import BWRun


def show_bubbles_2d(ax, data, bw, alpha_coefficient=1):
    A = bw.A
    mu = bw.mu
    L = bw.L
    n_obs = np.array(bw.n_obs)
    ax.cla()
    ax.scatter(data[:, 0], data[:, 1], s=5, color='#004cff',
               alpha=np.power(1 - bw.eps, np.arange(data.shape[0], 0, -1)))
    for n in reversed(np.arange(A.shape[0])):
        if n not in bw.dead_nodes:
            el = np.linalg.inv(L[n])
            sig = el.T @ el
            u, s, v = np.linalg.svd(sig)
            width, height = np.sqrt(s[0]) * 3, np.sqrt(s[1]) * 3
            angle = atan2(v[0, 1], v[0, 0]) * 360 / (2 * np.pi)
            el = Ellipse((mu[n, 0], mu[n, 1]), width, height, angle=angle, zorder=8)
            el.set_alpha(0.4 * alpha_coefficient)
            el.set_clip_box(ax.bbox)
            el.set_facecolor('#ed6713')
            ax.add_artist(el)
            d = min(width, height)
            ax.text(mu[n, 0] + .5 * d, mu[n, 1] + .5 * d, str(n), clip_on=True)
        else:
            el = np.linalg.inv(L[n])
            sig = el.T @ el
            u, s, v = np.linalg.svd(sig)
            width, height = np.sqrt(s[0]) * 3, np.sqrt(s[1]) * 3
            angle = atan2(v[0, 1], v[0, 0]) * 360 / (2 * np.pi)
            el = Ellipse((mu[n, 0], mu[n, 1]), width, height, angle=angle, zorder=8)
            el.set_alpha(0.05 * alpha_coefficient)
            el.set_clip_box(ax.bbox)
            el.set_facecolor('#000000')
            ax.add_artist(el)

    mask = np.ones(mu.shape[0], dtype=bool)
    mask[n_obs < .1] = False
    mask[bw.dead_nodes] = False
    ax.scatter(mu[mask, 0], mu[mask, 1], c='k', zorder=10)
    ax.scatter(data[0, 0], data[0, 1], color="#004cff", s=10)


def show_active_bubbles_2d(ax, data, bw):
    mu = bw.mu
    L = bw.L
    ax.cla()
    ax.scatter(data[:, 0], data[:, 1], s=5, color='#004cff',
               alpha=np.power(1 - bw.eps, np.arange(data.shape[0], 0, -1)))
    ax.scatter(data[-1, 0], data[-1, 1], s=10, color='red')

    to_draw = np.argsort(np.array(bw.alpha))[-3:]
    opacities = np.array(bw.alpha)[to_draw]
    opacities = opacities * .5 / opacities.max()

    for i, n in enumerate(to_draw):
        el = np.linalg.inv(L[n])
        sig = el.T @ el
        u, s, v = np.linalg.svd(sig)
        width, height = np.sqrt(s[0]) * 3, np.sqrt(s[1]) * 3
        angle = atan2(v[0, 1], v[0, 0]) * 360 / (2 * np.pi)
        el = Ellipse((mu[n, 0], mu[n, 1]), width, height, angle=angle, zorder=8)
        el.set_alpha(opacities[i])
        el.set_clip_box(ax.bbox)
        el.set_facecolor('#ed6713')
        ax.add_artist(el)
        d = min(width, height)
        ax.text(mu[n, 0] + .25 * d, mu[n, 1] + .25 * d, str(n), alpha=min(opacities[i] * 2, 1), clip_on=True)


# def br_plot_3d(br):
#     # TODO: make a plot_3d like above
#     fig = plt.figure()
#     ax = plt.axes(projection="3d")
#
#     # Set of all spherical angles to draw our ellipsoid
#     n_points = 10
#     theta = np.linspace(0, 2*np.pi, n_points)
#     phi = np.linspace(0, np.pi, n_points)
#
#     # Get the xyz points for plotting
#     # Cartesian coordinates that correspond to the spherical angles:
#     X = np.outer(np.cos(theta), np.sin(phi))
#     Y = np.outer(np.sin(theta), np.sin(phi)).flatten()
#     Z = np.outer(np.ones_like(theta), np.cos(phi)).flatten()
#     old_shape = X.shape
#     X = X.flatten()
#
#
#     s = np.load(br.file)
#     data = s['y'][0]
#
#     A = br.A
#     mu = br.mu
#     L = br.L
#     n_obs = br.n_obs
#
#     # TODO: make these not lists
#     pred = br.pred_list[:,0]
#     entropy = br.entropy_list[:,0]
#
#     ax.plot(data[:,0], data[:,1], data[:,2], color='gray', alpha=0.8)
#     for n in np.arange(A.shape[0]):
#         if n_obs[n] > 1e-4:
#             el = np.linalg.inv(L[n]).T
#             sig = el @ el.T
#             # Find and sort eigenvalues to correspond to the covariance matrix
#             eigvals, eigvecs = np.linalg.eigh(sig)
#             idx = np.sum(sig,axis=0).argsort()
#             eigvals_temp = eigvals[idx]
#             idx = eigvals_temp.argsort()
#             eigvals = eigvals[idx]
#             eigvecs = eigvecs[:,idx]
#
#             # Width, height and depth of ellipsoid
#             nstd = 3
#             rx, ry, rz = nstd * np.sqrt(eigvals)
#
#             # Rotate ellipsoid for off axis alignment
#             a,b,c = np.matmul(eigvecs, np.array([X*rx,Y*ry,Z*rz]))
#             a,b,c = a.reshape(old_shape), b.reshape(old_shape), c.reshape(old_shape)
#
#             # Add in offsets for the mean
#             a = a + mu[n,0]
#             b = b + mu[n,1]
#             c = c + mu[n,2]
#
#             ax.plot_surface(a, b, c, color='#ff4400', alpha=0.6)
#
#     ax.view_init(40,23)
#
#     mask = np.ones(mu.shape[0], dtype=bool)
#     mask[n_obs<1e-4] = False
#     ax.scatter(mu[mask,0], mu[mask,1], mu[mask,2], c='k' , zorder=10)
#
#     ax.set_xticks([200, 600, 1000, 1400])
#     ax.set_yticks([-20, -10, 0, 10])
#     ax.set_zticks([-1400, -1000, -600, -200])
#     in1, in2 = 0, 1
#     ax.text(in1, in2, 100, s='b', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
#     plt.show()

def show_A(ax, bw):
    ax.cla()
    ims = ax.imshow(bw.A, aspect='equal', interpolation='nearest')

    ax.set_title("Transition Matrix (A)")
    ax.set_xlabel("To")
    ax.set_ylabel("From")

    ax.set_xticks(np.arange(bw.N))
    live_nodes = [x for x in np.arange(bw.N) if x not in bw.dead_nodes]
    ax.set_yticks(live_nodes)


# def show_Ct_y(ax, regressor):
#     old_ylim = ax.get_ylim()
#     ax.cla()
#     ax.plot(regressor.Ct_y, '.-')
#     ax.set_title("Ct_y")
#
#     new_ylim = ax.get_ylim()
#     ax.set_ylim([min(old_ylim[0], new_ylim[0]), max(old_ylim[1], new_ylim[1])])


def show_alpha(ax, br, offset=0):
    ax.cla()
    ims = ax.imshow(np.array(br.alpha_history[0][-19:] + [br.bw.alpha]).T, aspect='auto', interpolation='nearest')

    ax.set_title("State Estimate ($\\alpha$)")
    live_nodes = [x for x in np.arange(br.bw.N) if x not in br.bw.dead_nodes]
    ax.set_yticks(live_nodes)
    ax.set_ylabel("bubble")
    ax.set_xlabel("steps (ago)")
    # ax.set_xticks([0.5,5,10,15,20])
    # ax.set_xticklabels([-20, -15, -10, -5, 0])


def show_behavior_variables(ax, br, beh):
    # todo: this function is fragile, maybe delete it?
    ax.cla()
    br: bubblewrap.bw_run.BWRun
    ax.plot(br.behavior_pred_history[0][-20:])
    ax.plot(beh[-20:])
    ax.set_title("Behavior prediction")


def show_A_eigenspectrum(ax, bw):
    ax.cla()
    eig = np.sort(np.linalg.eigvals(bw.A))[::-1]
    ax.plot(eig, '.')
    ax.set_title("Eigenspectrum of A")
    ax.set_ylim([0, 1])


def _mean_distance(data, shift=1):
    x = data - data.mean(axis=0)
    T = x.shape[0]
    differences = x[0:T - shift] - x[shift:T]
    distances = np.linalg.norm(differences, axis=1)
    return distances.mean()


def show_data_distance(ax, d, max_step=50):
    """like a distance autocovariance function"""

    old_ylim = ax.get_ylim()
    ax.cla()
    if d.shape[0] > 10:
        shifts = np.arange(0, min(d.shape[0] // 2, max_step))
        distances = [_mean_distance(d, shift) for shift in shifts]
        ax.plot(shifts, distances)
    ax.set_xlim([0, max_step])
    new_ylim = ax.get_ylim()
    ax.set_ylim([0, max(old_ylim[1], new_ylim[1])])
    ax.set_title(f"dataset distances")
    ax.set_xlabel("offset")
    ax.set_ylabel("distance")


def show_nstep_pred_pdf(ax, br, other_axis, fig, offset=1):
    """
    the other_axis is supposed to be something showing the bubbles, so they line up
    """
    if ax.collections:
        old_vmax = ax.collections[-3].colorbar.vmax
        old_vmin = ax.collections[-3].colorbar.vmin
        ax.collections[-3].colorbar.remove()
    ax.cla()

    other_axis: plt.Axes
    xlim = other_axis.get_xlim()
    ylim = other_axis.get_ylim()

    bw = br.bw

    density = 50
    x_bins = np.linspace(*xlim, density + 1)
    y_bins = np.linspace(*ylim, density + 1)
    pdf = np.zeros(shape=(density, density))
    for i in range(density):
        for j in range(density):
            x = np.array([x_bins[i] + x_bins[i + 1], y_bins[j] + y_bins[j + 1]]) / 2
            b_values = bw.logB_jax(x, bw.mu, bw.L, bw.L_diag)
            pdf[i, j] = bw.alpha @ np.linalg.matrix_power(bw.A, offset) @ np.exp(b_values)

    # these might control the colors
    # cmesh = ax.pcolormesh(x_bins,y_bins,pdf.T, vmin=min(vmin, pdf.min()), vmax=max(vmax, pdf.max()))
    # cmesh = ax.pcolormesh(x_bins,y_bins,pdf.T, vmin=0, vmax=0.03) #log, vmin=-15, vmax=-5

    cmesh = ax.pcolormesh(x_bins, y_bins, pdf.T)
    fig.colorbar(cmesh)

    current_location = br.obs_ds.get_atemporal_data_point(0)
    offset_location = br.obs_ds.get_atemporal_data_point(offset)

    ax.scatter(offset_location[0], offset_location[1], c='white')

    ax.scatter(current_location[0], current_location[1], c='red')

    ax.set_title(f"{offset}-step pred.")


def show_alphas_given_regression_value(ax, br, behavior_value, step, history_length=50, hist_axis=None):
    br: BWRun
    ax: plt.Axes
    obs_hist, beh_hist = br.data_source.get_history()
    shorter = min(len(beh_hist), len(br.alpha_history))
    beh_hist = beh_hist[-shorter:]
    a_hist = np.array(br.alpha_history[-shorter:])
    mask = np.squeeze(beh_hist == behavior_value)
    to_show = a_hist[mask]
    to_show = to_show[-history_length:]

    ax.imshow(to_show, interpolation='nearest')

    ax.set_title(f"$\\alpha$ given beh=={behavior_value}")
    ax.set_xlabel("$\\alpha$ index")
    if hist_axis is not None:
        hist_axis.cla()
        hist_axis: plt.Axes
        hist_axis.bar(np.arange(to_show.shape[1]), to_show.sum(axis=0) / to_show.sum())


# def show_w_sideways(ax, bw, current_behavior):
#     ax.cla()
#     w = np.array(bw.D @ bw.Ct_y)
#     w[bw.dead_nodes] = 0
#
#     a = np.array(bw.alpha)
#     a = a / np.max(a)
#     ax.plot(w, np.arange(w.size), alpha=0.25)
#     ax.scatter(w, np.arange(w.size), alpha=a, c="C0")
#     ylim = ax.get_ylim()
#     ax.vlines(current_behavior[0], alpha=.5, ymin=ylim[0], ymax=ylim[1], colors="C1" )
#     ax.set_ylabel("bubble #")
#     ax.set_xlabel("weight magnitude")
#     ax.set_title(r"Weights (times $\alpha$)")
#     ax.set_xlim([-21, 21])


def _one_sided_ewma(data, com=100):
    import pandas as pd
    return pd.DataFrame(data=dict(data=data)).ewm(com).mean()["data"]


def _deduce_bw_parameters(bw):
    bw: bubblewrap.Bubblewrap
    return dict(dim=bw.d,
                num=bw.N,
                seed=bw.seed,
                M=bw.M,
                step=bw.step,
                lam=bw.lam_0,
                eps=bw.eps,
                nu=bw.nu,
                B_thresh=bw.B_thresh,
                n_thresh=bw.n_thresh,
                batch=bw.batch,
                batch_size=bw.batch_size,
                go_fast=bw.go_fast,
                copy_row_on_teleport=bw.copy_row_on_teleport,
                num_grad_q=bw.num_grad_q,
                )


def compare_metrics(brs, offset, colors=None, smoothing_scale=50, show_legend=True, show_title=True):
    ps = [_deduce_bw_parameters(br.bw) for br in brs]
    keys = set([leaf for tree in ps for leaf in tree.keys()])
    keep_keys = []
    for key in keys:
        values = [d.get(key) for d in ps]
        if not all([values[0] == v for v in values]):
            keep_keys.append(key)
    to_print = []
    for key in keep_keys:
        to_print.append(f"{key}: {[p.get(key) for p in ps]}")

    fig, ax = plt.subplots(figsize=(12, 5), nrows=3, ncols=1, sharex='all', layout='tight')
    to_write = [[], [], []]

    n = 0
    for idx, br in enumerate(brs):
        br: bubblewrap.bw_run.BWRun
        n = max(n, br.entropy_history[offset].shape[0])

        xlim = n * np.array([-0.01, 1.07])

        predictions = br.prediction_history[offset]
        smoothed_predictions = _one_sided_ewma(predictions, smoothing_scale)

        c = 'black'
        if colors:
            c = colors[idx]
        ax[0].plot(predictions, alpha=0.25, color=c)
        ax[0].plot(smoothed_predictions, color=c, label=br.pickle_file.split("/")[-1].split(".")[0].split("_")[-1])
        ax[0].tick_params(axis='y')
        # todo: bring this back
        ax[0].set_ylabel('prediction')
        to_write[0].append((idx, f"{predictions[n // 2:].mean():.3f}", dict(color=c)))

        entropy = br.entropy_history[offset]
        smoothed_entropy = _one_sided_ewma(entropy, smoothing_scale)

        c = 'black'
        if colors:
            c = colors[idx]
        ax[1].plot(entropy, color=c, alpha=0.25)

        ax[1].plot(smoothed_entropy, color=c)
        max_entropy = np.log2(br.bw.N)
        ax[1].plot([0, entropy.shape[0]], [max_entropy, ] * 2, 'k--')

        ax[1].tick_params(axis='y')
        ax[1].set_ylabel('entropy')
        to_write[1].append((idx, f"{entropy[n // 2:].mean():.3f}", dict(color=c)))

        beh_error = np.squeeze(br.behavior_error_history[offset] ** 2)
        c = 'black'
        if colors:
            c = colors[idx]
        ax[2].plot(beh_error, color=c)
        ax[2].set_ylabel('behavior mse')
        ax[2].tick_params(axis='y')
        to_write[2].append((idx, f"{beh_error[n // 2:].mean():.3f}", dict(color=c)))

    for i, l in enumerate(to_write):
        ylim = ax[i].get_ylim()
        yrange = ylim[1] - ylim[0]
        for idx, text, kw in l:
            ax[i].text(n * 1.01, ylim[1] - (idx + 1) * yrange / 7, text, clip_on=True, **kw)

    xticks = list(ax[2].get_xticks())
    xticks.append(n // 2)
    ax[2].set_xticks(xticks)
    ax[0].set_xlim(xlim)
    if show_title:
        ax[0].set_title(" ".join(to_print))
    else:
        print(to_print)
    if show_legend:
        ax[0].legend(loc="lower right")
