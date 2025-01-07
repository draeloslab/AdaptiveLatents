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
