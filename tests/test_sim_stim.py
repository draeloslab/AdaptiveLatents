import numpy as np
import pytest
import warnings
from itertools import cycle

from adaptive_latents.sim_stim import SimulatedStimAdder, make_sr
from adaptive_latents.stim_regressor import StimRegressor
from adaptive_latents.stim_designer import StimDesigner
from adaptive_latents import datasets


longrun = pytest.mark.skipif("not config.getoption('longrun')")

@pytest.fixture
def input_array(rng):
    return datasets.Odoherty21Dataset().neural_data.slice_by_time(slice(None, 30))


@pytest.fixture
def input_array_long(rng):
    return datasets.Odoherty21Dataset().neural_data.slice_by_time(slice(None, None))


class TestSimulatedStimAdder:
    def test_run_for_X_no_delay(self):
        """Test run_for_X applies stimulation immediately with no delay."""
        adder = SimulatedStimAdder(stim_time_delay=0, decay=0.8)

        # Register a stim
        stim_result = np.array([10.0, 0.0, 0.0])
        adder.register_stim(stim_result)

        # First call should add the full stim
        data = np.array([1.0, 1.0, 1.0])
        result = adder.run_for_X(data)
        adder.register_stim(0)
        expected = np.array([11.0, 1.0, 1.0])
        assert np.allclose(result, expected)

        # Second call should apply decay
        data2 = np.array([1.0, 1.0, 1.0])
        result2 = adder.run_for_X(data2)
        adder.register_stim(0)
        expected2 = np.array([9.0, 1.0, 1.0])  # 1 + 10*0.8
        assert np.allclose(result2, expected2)

    def test_run_for_X_with_delay(self):
        """Test run_for_X applies stimulation after delay."""
        adder = SimulatedStimAdder(stim_time_delay=2, decay=1.0)

        # Register a stim
        stim_result = np.array([5.0, 0.0, 0.0])
        adder.register_stim(stim_result)

        data = np.array([1.0, 1.0, 1.0])

        # First two calls should not add stim (delay = 2)
        result1 = adder.run_for_X(data.copy())
        assert np.allclose(result1, data)

        result2 = adder.run_for_X(data.copy())
        assert np.allclose(result2, data)

        # Third call should add the stim
        result3 = adder.run_for_X(data.copy())
        expected3 = np.array([6.0, 1.0, 1.0])
        assert np.allclose(result3, expected3)

    def test_exponential_decay(self):
        """Test exponential decay over multiple timesteps."""
        adder = SimulatedStimAdder(stim_time_delay=0, decay=0.5)

        # Register a stim
        stim_result = np.array([100.0, 0.0, 0.0])
        adder.register_stim(stim_result)

        data = np.array([0.0, 0.0, 0.0])

        # Track the decay over time
        results = []
        for _ in range(5):
            result = adder.run_for_X(data.copy())
            adder.register_stim(0)
            results.append(result[0])

        # Should decay as: 100, 50, 25, 12.5, 6.25
        expected = [100.0, 50.0, 25.0, 12.5, 6.25]
        assert np.allclose(results, expected)

    def test_true_stim_result_identity(self):
        """Test true_stim_result with identity transformation."""
        adder = SimulatedStimAdder(true_S='identity')

        stim = np.array([1.0, 2.0, 3.0])
        result = adder.true_stim_result(stim)

        assert np.allclose(result, stim)

    def test_true_stim_result_flip(self):
        """Test true_stim_result with flip transformation."""
        adder = SimulatedStimAdder(true_S='flip')

        # Create a simple projection matrix (2D subspace of 3D space)
        proj_matrix = np.array([[1.0, 0.0],
                                [0.0, 1.0],
                                [0.0, 0.0]])

        # Stim in the subspace
        stim = np.array([2.0, 3.0, 0.0])
        result = adder.true_stim_result(stim, proj_matrix)

        # Should flip the in-space components: [3.0, 2.0, 0.0]
        expected = np.array([3.0, 2.0, 0.0])
        assert np.allclose(result, expected)

    def test_true_stim_result_flip_with_out_of_space(self):
        """Test flip transformation preserves out-of-space components."""
        adder = SimulatedStimAdder(true_S='flip')

        # 2D subspace of 3D space
        proj_matrix = np.array([[1.0, 0.0],
                                [0.0, 1.0],
                                [0.0, 0.0]])

        # Stim with both in-space and out-of-space components
        stim = np.array([2.0, 3.0, 5.0])
        result = adder.true_stim_result(stim, proj_matrix)

        # In-space should flip, out-of-space should stay: [3.0, 2.0, 5.0]
        expected = np.array([3.0, 2.0, 5.0])
        assert np.allclose(result, expected)

    def test_true_stim_result_high_d_permuted(self):
        """Test high_d_permuted transformation is deterministic."""
        adder = SimulatedStimAdder(true_S='high_d_permuted', static_S_seed=42)

        stim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result1 = adder.true_stim_result(stim.copy())
        result2 = adder.true_stim_result(stim.copy())

        # Should be deterministic
        assert np.allclose(result1, result2)

        # Should be a permutation (same elements)
        assert np.allclose(sorted(result1), sorted(stim))

        # Should not be identity
        assert not np.allclose(result1, stim)


class TestMakeSRBasics:
    """Basic integration tests for make_sr function."""

    def test_basic_run_completes(self, input_array, rng):
        """Test basic make_sr run completes without errors."""
        sr, stim_designer, log = make_sr(
            input_array=input_array,
            rng=rng,
            exit_time=20,
            stim_rate=2,
            stim_timing_method='isi',
        )

        # Check return types
        assert isinstance(sr, StimRegressor)
        assert isinstance(stim_designer, StimDesigner)
        assert isinstance(log, dict)

    def test_log_contains_expected_keys(self, input_array, rng):
        """Test log dictionary contains all expected keys."""
        sr, stim_designer, log = make_sr(
            input_array=input_array,
            rng=rng,
            exit_time=20,
            stim_rate=2,
            prosvd_k=5,
            stim_timing_method='isi',
        )

        expected_keys = [
            'high_d_stims',
            'high_d_without_stim',
            'high_d_with_stim',
            'latents',
            'stim_intended_samples',
            'stims',
            'timing_log'
        ]

        for key in expected_keys:
            assert key in log, f"Missing key: {key}"


        # High-D data should match input dimensions
        input_dim = input_array.shape[1]
        assert log['high_d_stims'].shape[1] == input_dim
        assert log['high_d_without_stim'].shape[1] == input_dim
        assert log['high_d_with_stim'].shape[1] == input_dim

        # Latents should match prosvd_k
        assert log['latents'].shape[1] == 5

        reconstructed = log['high_d_without_stim'] + log['high_d_stims']
        assert np.allclose(log['high_d_with_stim'], reconstructed)

        assert log['stims'].shape[0] > 0
        assert not (log['high_d_stims'] == 0).all()



        timing_log = log['timing_log']

        # Check timing attributes exist
        assert hasattr(timing_log, 'init_time')
        assert hasattr(timing_log, 'loop_time')
        assert hasattr(timing_log, 'stim_design')
        assert hasattr(timing_log, 'dimension_reduction')
        assert hasattr(timing_log, 'sr_update')
        assert hasattr(timing_log, 'per_loop')
        assert hasattr(timing_log, 'in_sim_time')

        # Lists should have reasonable lengths
        n_steps = len(timing_log.in_sim_time)
        assert n_steps > 0
        assert len(timing_log.stim_design) == n_steps
        assert len(timing_log.dimension_reduction) == n_steps


class TestMakeSRStimulationTiming:
    """Tests for different stimulation timing methods."""

    def test_stim_rate(self, input_array, rng):
        """Test stimulation with stim_rate parameter."""
        sr, stim_designer, log = make_sr(
            input_array=input_array,
            rng=rng,
            exit_time=20,
            stim_rate=2,  # One stim every 2 seconds
            stim_timing_method='isi',
            initial_nostim_period=2,
        )

        # Should have approximately exit_time / stim_rate stims
        n_stims = log['stims'].shape[0]
        expected_stims = (20 - 2) / 2  # accounting for initial_nostim_period
        assert n_stims >= expected_stims * 0.8  # Allow some tolerance

    def test_isi_generator(self, input_array, rng):
        """Test stimulation with isi_generator."""
        sr, stim_designer, log = make_sr(
            input_array=input_array,
            rng=rng,
            exit_time=20,
            isi_generator=cycle([3.0]),  # One stim every 3 seconds
            stim_timing_method='isi',
            initial_nostim_period=2,
            stim_rate=None,  # todo: it would be nice if this weren't necessary to allow stim_timing_method='isi'
        )

        # Should have stims delivered
        assert log['stims'].shape[0] > 0

    def test_random_timing(self, input_array, rng):
        """Test random stimulation timing."""
        sr, stim_designer, log = make_sr(
            input_array=input_array,
            rng=rng,
            exit_time=20,
            stim_rate=2,
            stim_timing_method='random',
            initial_nostim_period=2,
        )

        # Should still deliver stims
        assert log['stims'].shape[0] > 0


class TestMakeSRDimensionReduction:
    """Tests for different dimension reduction methods."""

    def test_prosvd_only(self, input_array, rng):
        """Test with proSVD only (default)."""
        sr, stim_designer, log = make_sr(
            input_array=input_array,
            rng=rng,
            exit_time=20,
            last_dim_red='prosvd',
            prosvd_k=7,
        )

        # Latents should have prosvd_k dimensions
        assert log['latents'].shape[1] == 7

    def test_sjpca(self, input_array, rng):
        """Test with sjPCA dimension reduction."""
        sr, stim_designer, log = make_sr(
            input_array=input_array,
            rng=rng,
            exit_time=20,
            last_dim_red='sjpca',
            prosvd_k=5,
        )

        # Should complete without errors
        assert log['latents'].shape[0] > 0

    def test_mmica(self, input_array, rng):
        """Test with mmICA dimension reduction."""
        sr, stim_designer, log = make_sr(
            input_array=input_array,
            rng=rng,
            exit_time=20,
            last_dim_red='mmica',
            prosvd_k=5,
        )

        # Should complete without errors
        assert log['latents'].shape[0] > 0


class TestMakeSRTrueSTransforms:
    """Tests for different true_S transformation modes."""

    def test_identity_transform(self, input_array, rng):
        """Test with identity transformation."""
        sr, stim_designer, log = make_sr(
            input_array=input_array,
            rng=rng,
            exit_time=20,
            true_S='identity',
            stim_rate=2,
        )

        # Should complete and deliver stims
        assert log['stims'].shape[0] > 0

    def test_flip_transform(self, input_array, rng):
        """Test with flip transformation."""
        sr, stim_designer, log = make_sr(
            input_array=input_array,
            rng=rng,
            exit_time=20,
            true_S='flip',
            stim_rate=2,
        )

        # Should complete and deliver stims
        assert log['stims'].shape[0] > 0

    def test_high_d_permuted_transform(self, input_array, rng):
        """Test with high_d_permuted transformation."""
        sr, stim_designer, log = make_sr(
            input_array=input_array,
            rng=rng,
            exit_time=20,
            true_S='high_d_permuted',
            stim_rate=2,
        )

        # Should complete and deliver stims
        assert log['stims'].shape[0] > 0


class TestMakeSRDelays:
    """Tests for stimulation delays."""

    def test_stim_time_delay(self, input_array, rng):
        """Test with stim_time_delay."""
        sr, stim_designer, log = make_sr(
            input_array=input_array,
            rng=rng,
            exit_time=20,
            stim_time_delay=3,
            stim_rate=2,
        )

        # Should complete successfully
        assert log['stims'].shape[0] > 0

    def test_regressor_stim_delay(self, input_array, rng):
        """Test with regressor_stim_delay."""
        sr, stim_designer, log = make_sr(
            input_array=input_array,
            rng=rng,
            exit_time=20,
            regressor_stim_delay=2 * input_array.dt,
            stim_rate=2,
        )

        # Should complete successfully
        assert log['stims'].shape[0] > 0

    def test_both_delays(self, input_array, rng):
        """Test with both stim_time_delay and regressor_stim_delay."""
        sr, stim_designer, log = make_sr(
            input_array=input_array,
            rng=rng,
            exit_time=20,
            stim_time_delay=2,
            regressor_stim_delay=3 * input_array.dt,
            stim_rate=2,
        )

        # Should complete successfully
        assert log['stims'].shape[0] > 0


class TestMakeSRUToSModels:
    """Tests for different u_to_s model types."""

    def test_identity_u_to_s(self, input_array, rng):
        """Test with identity u_to_s model (open loop)."""
        sr, stim_designer, log = make_sr(
            input_array=input_array,
            rng=rng,
            exit_time=20,
            u_to_s_model_type='identity',
            stim_rate=2,
        )

        assert log['stims'].shape[0] > 0
        # Check that stim_designer logged appropriately
        assert len(stim_designer.log) > 0


class TestMakeSRDeterminism:
    """Tests for deterministic behavior."""

    def test_same_seed_same_results(self, input_array):
        """Test that same RNG seed produces same results."""
        rng1 = np.random.default_rng(42)
        sr1, stim_designer1, log1 = make_sr(
            input_array=input_array,
            rng=rng1,
            exit_time=10,
            stim_rate=2,
        )

        rng2 = np.random.default_rng(42)
        sr2, stim_designer2, log2 = make_sr(
            input_array=input_array,
            rng=rng2,
            exit_time=10,
            stim_rate=2,
        )

        # Results should be identical
        assert np.allclose(log1['latents'], log2['latents'])
        assert np.allclose(log1['high_d_stims'], log2['high_d_stims'])
        assert log1['stims'].shape[0] == log2['stims'].shape[0]


@longrun
class TestMakeSRLongRun:
    """Long-running tests requiring more computation time."""

    def test_closed_loop_kernel_regressed(self, input_array_long, rng):
        """Test closed-loop with kernel_regressed u_to_s model."""
        sr, stim_designer, log = make_sr(
            input_array=input_array_long,
            rng=rng,
            exit_time=60,
            u_to_s_model_type='kernel_regressed',
            stim_rate=2,
            stim_timing_method='isi',
            initial_nostim_period=5,
        )

        # Should complete successfully
        assert log['stims'].shape[0] > 0

        # Check that stim_designer log has observed_s_hat entries
        logs_with_observed = [l for l in stim_designer.log if 'observed_s_hat' in l]
        assert len(logs_with_observed) > 0

        # Check stim_reg was actually trained
        assert sr.stim_reg.n_observed > 0