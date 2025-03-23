import jax
import numpy as np
import pytest

import adaptive_latents
from adaptive_latents import ArrayWithTime


class TestJaxEnvironment:
    def test_can_use_configured_backend(self):
        # note that this does not check that both backends are possible
        from jax.extend.backend import get_backend
        assert get_backend().platform == adaptive_latents.CONFIG.jax_platform_name

    def test_can_use_float64(self):
        jax.config.update('jax_enable_x64', True)  # this line is for documentation, the real line is in the config load
        x = jax.random.uniform(jax.random.key(0), (1,), dtype=jax.numpy.float64)
        assert x.dtype == jax.numpy.float64

    # for documentation purposes
    """
    def test_can_use_float32(self):
        import jax
        # jax.config.update('jax_enable_x64', False)
        x = jax.random.uniform(jax.random.key(0), (1,), dtype=jax.numpy.float64)
        assert x.dtype != jax.numpy.float64
    """


def test_utils_run(rng):
    A = rng.normal(size=(200, 10))
    t = np.arange(A.shape[0])

    t, old_t = np.linspace(0, 10), t
    adaptive_latents.utils.resample_matched_timeseries(A, old_t, t)
    A, t = adaptive_latents.utils.clip(A, t)


def test_cache_works(rng, tmp_path):
    should_fail = False
    def f(n):
        assert not should_fail
        if n < 1:
            return 0
        elif n == 1:
            return 1

        return np.array(f(n-1) + f(n-2))

    cached_f = adaptive_latents.utils.save_to_cache("fibonacci_test", tmp_path, override_config_and_cache=True)(f)

    assert cached_f(6) == 8

    should_fail = True  # changes the value above
    with pytest.raises(AssertionError):
        f(6)

    assert cached_f(6) == 8


# TODO:
#  array shapes are correct for 1d output
#  test different regressors work together
#  test_can_save_and_reload
#  test_nsteps_inbwrun_works_correctly
#  also should make the timing of logs more clear
#  make sure the config in-file defaults equal the repo defaults

@pytest.mark.parametrize("a,b,expected", [
    (
        ArrayWithTime([1,2,3], [1,2,3]),
        ArrayWithTime([1,2,3], [1,2,3]),
        ArrayWithTime([0,0,0], [1,2,3])
    ),
    (
        ArrayWithTime([2, 3], [2, 3]),
        ArrayWithTime([1, 2, 3], [1, 2, 3]),
        ArrayWithTime([0, 0], [2, 3])
    ),
    (
            ArrayWithTime([2, 3, 4], [2, 3, 4]),
            ArrayWithTime([1, 2, 3], [1, 2, 3]),
            ArrayWithTime([0, 0], [2, 3])
    ),
    (
        ArrayWithTime([0,1,2,3,4], [0,1,2,3,4]),
        ArrayWithTime([1,2,3], [1,2,3]),
        ArrayWithTime([0,0,0], [1,2,3])
    ),
])
def test_array_subtraction_works(a,b,expected):
    for aa, bb in [(a,b), [b,a]]:
        diff = ArrayWithTime.subtract_aligned_indices(aa, bb)
        assert (diff == expected).all()
        assert (diff.t == expected.t).all()