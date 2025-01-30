import numpy as np
import tensortools as tt
from tensortools.cpwarp import fit_shifted_cp

def organize_neural_data_by_stimulation(d, peri_stim_window=None):
    if peri_stim_window is None:
        peri_stim_window = [0, np.diff(d.opto_stimulations['sample']).min()]

    peri_stim_recordings = []
    d.opto_stimulations.sort_values(by='time')
    for stim_sample in d.opto_stimulations['sample']:
        samples = np.arange(d.neural_data.shape[0])
        s = (stim_sample + peri_stim_window[0] <= samples) & (samples < stim_sample + peri_stim_window[1])
        if not s[-1]:
            peri_stim_recordings.append(d.neural_data[s, :])

    return np.array(peri_stim_recordings)


def make_responses(d, non_nan=False):
    peri_stim_recordings = organize_neural_data_by_stimulation(d)
    responses = peri_stim_recordings - peri_stim_recordings[:, 0:1, :]
    if non_nan:
        responses = responses[..., :d.n_neurons_in_optical]
    return responses


def reshape_by_group(a, d):
    responses_by_group = []
    groups = []
    for neuron, psth in zip(d.opto_stimulations.target_neuron, a):
        if not groups or groups[-1] != neuron:
            if groups:
                responses_by_group[-1] = np.array(responses_by_group[-1])
            responses_by_group.append([])
            groups.append(neuron)
        responses_by_group[-1].append(psth)

    responses_by_group[-1] = np.array(responses_by_group[-1])

    return responses_by_group


def find_decompositions(non_nan_responses, n_restarts=200):
    models = []
    for _ in range(n_restarts):
        try:
            m = tt.cpwarp.fit_shifted_cp(
                non_nan_responses.transpose(2, 0, 1),
                1,
                max_iter=1000,
                boundary="edge",
                max_shift_axis0=None,
                max_shift_axis1=.3,
                u_nonneg=True,  # neurons
                v_nonneg=True,  # trials
            )
            models.append(m)
        except ZeroDivisionError:
            pass

    models.sort(key=lambda m: m.loss_hist[-1])
    return models


def compare_rows(a, method='angles'):
    comparison_matrix = np.zeros([len(a), len(a)]) * np.nan
    for i in range(comparison_matrix.shape[0]):
        for j in range(comparison_matrix.shape[1]):
            ai_hat = a[i] / np.linalg.norm(a[i])
            aj_hat = a[j] / np.linalg.norm(a[j])
            if method == 'angles':
                comparison_matrix[i, j] = np.arccos(np.clip(ai_hat @ aj_hat, -1, 1)) * 180 / np.pi
            elif method == 'distances':
                comparison_matrix[i, j] = np.linalg.norm(a[i] - a[j])
            elif method == 'norm_distances':
                comparison_matrix[i, j] = np.linalg.norm(ai_hat - aj_hat)
            else:
                comparison_matrix[i, j] = method(a[i], a[j])
    return comparison_matrix


def plot_comparison(ax, comparison_matrix, vmin=None, vmax=None, ax_colorbar=None, group_sizes=None, group_names=None, fig=None):
    ax.cla()
    if vmin is None:
        vmin = comparison_matrix[comparison_matrix > comparison_matrix.min()].min()
    im = ax.matshow(comparison_matrix, vmin=vmin, vmax=vmax)
    if ax_colorbar is not None and fig is not None:
        fig.colorbar(mappable=im, cax=ax_colorbar)

    if group_sizes is not None:
        group_edges = np.cumsum(group_sizes) - .5
        for boundary in group_edges:
            lw = .5
            ax.axvline(boundary, color='w', lw=lw)
            ax.axhline(boundary, color='w', lw=lw)
        ax.set_xticks([e - s / 2 for s, e in zip(group_sizes, group_edges)])
        if group_names is None:
            group_names = [f"{chr(65 + g)}" for g in range(len(group_sizes))]
        ax.set_xticklabels(group_names)
        ax.set_xlabel('group #')
        ax.set_ylabel('stim #')


def plot_per_neuron(ax, fg_s, d, bg_s=1, bg_c='k', bg_alpha=.1, fg_c='red'):
    fg_s = np.squeeze(fg_s)
    neuron_locations = d.neuron_locations
    if fg_s.shape[0] < neuron_locations.shape[0]:
        neuron_locations = neuron_locations[:fg_s.shape[0]]
    ax.scatter(neuron_locations[:,1], neuron_locations[:,0], s=bg_s, color=bg_c, alpha=bg_alpha)
    ax.scatter(neuron_locations[:,1], neuron_locations[:,0], s=fg_s, color=fg_c)
    ax.axis('equal')
    ax.axis('off')
