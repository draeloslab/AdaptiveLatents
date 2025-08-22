import sim_stim_refactored
import sys
sys.path.append('/home/jgould/Documents/AdaptiveLatents/workspace/neurips_2025/simulation_plots')
import sim_stim as sim_stim_old
import pytest
import numpy as np
import adaptive_latents as al
import copy
import jaxlib
import matplotlib.pyplot as plt
import inspect

# pytest test_consistency.py --timeout=10000 --pdb

@pytest.mark.parametrize('comparison_preset', [
    'optim_col_vs_rand_with_high_d_rand',
    'optim_col_vs_rand',
    'pred methods',
    'optim_open_vs_closed_toy',
    'optim_open_vs_closed',
    # 'delay-table',
    'visualization',
    'default',
    ])
def test_consistency(comparison_preset, test_old_consistent=False):
    data = al.datasets.Odoherty21Dataset().neural_data
    rng = np.random.default_rng(0)
    preset = sim_stim_old.get_presets(comparison_preset)
    for k, v in preset.items():
        print(k, v)
        v = v | {'exit_time': 41}
        old_result_1 = sim_stim_old.make_sr(data, copy.deepcopy(rng), **v)
        if test_old_consistent:
            old_result_2 = sim_stim_old.make_sr(data, copy.deepcopy(rng), **v)
        new_sr, new_stim_designer, new_log = sim_stim_refactored.make_sr(data, copy.deepcopy(rng), **v)

        for key in ['high_d_without_stim', 'stim_intended_samples', 'high_d_stims', 'latents']:
            if test_old_consistent:
                old_result_2_value = old_result_2.log.pop(key)
                assert (old_result_1.log[key] == old_result_2_value).all(), key
                assert (old_result_1.log[key].t == old_result_2_value.t).all(), key

            old_log_value = old_result_1.log.pop(key)
            assert np.allclose(new_log[key], old_log_value), key
            assert (new_log[key].t == old_log_value.t).all(), key

        if test_old_consistent:
            compare_stim_designer_logs(old_result_1.stim_designer.log, old_result_2.stim_designer.log)
        compare_stim_designer_logs(new_stim_designer.log, old_result_1.stim_designer.log)



def compare_stim_designer_logs(a, b):
    for i, (n, o) in enumerate(zip(a, b)):
        assert (keys:=set(n.keys()) - {'optimization_time'}) == set(o.keys()- {'optimization_time'}), set(n.keys()).symmetric_difference(set(o.keys()))

        for key in keys:
            if isinstance(n[key], np.ndarray) or isinstance(n[key], jaxlib.xla_extension.ArrayImpl):
                assert np.array_equal(n[key], o[key], equal_nan=True), key
                if isinstance(n[key], al.ArrayWithTime):
                    assert np.array_equal(n[key].t, o[key].t, equal_nan=True), key
            elif isinstance(n[key], al.regressions.BaseKernelRegressor):
                assert (n[key].history is None and o[key].history is None) or np.array_equal(n[key].history, o[key].history, equal_nan=True), key
            else:
                assert n[key] == o[key], key
