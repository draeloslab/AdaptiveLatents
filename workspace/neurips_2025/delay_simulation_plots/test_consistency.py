import sim_stim_refactored
import sys
sys.path.append('/home/jgould/Documents/AdaptiveLatents/workspace/neurips_2025/simulation_plots')
import sim_stim as sim_stim_old
import pytest
import numpy as np
import adaptive_latents as al
import copy
import matplotlib.pyplot as plt
import inspect

@pytest.mark.parametrize('comparison_preset', [
        'pred methods',
        # 'optim_col_vs_rand',
        # 'optim_col_vs_rand_with_high_d_rand',
        # 'optim_open_vs_closed',
        # 'optim_open_vs_closed_toy',
        # 'delay-table',
        # 'default',
        # 'visualization',
    ])
def test_consistency(comparison_preset, show_plots=True):

    data = al.datasets.Zong22Dataset().neural_data
    rng = np.random.default_rng()
    preset = sim_stim_old.get_presets(comparison_preset)
    for k, v in preset.items():
        new_sr, new_stim_designer, new_log = sim_stim_refactored.make_sr(data, copy.deepcopy(rng), **v)
        old_result = sim_stim_old.make_sr(data, copy.deepcopy(rng), **v)

        if show_plots:
            fig, ax = plt.subplots()
            assert (new_log['latents'].t == old_result.log['latents'].t).all()
            ax.plot(new_log['latents'].t, new_log['latents'] - old_result.log['latents'])
            plt.show()

        assert (new_log['latents'] == old_result.log['latents']).all()
        assert new_sr.log == old_result.log
        assert new_stim_designer == old_result.stim_designer


    new_sig = inspect.signature(sim_stim_refactored.make_sr)
    old_sig = inspect.signature(sim_stim_old.make_sr)
    assert new_sig.parameters == old_sig.parameters