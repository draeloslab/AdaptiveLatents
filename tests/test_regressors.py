from adaptive_latents.regressions import NearestNeighborRegressor, SymmetricNoisyRegressor, WindowRegressor, VanillaOnlineRegressor, AutoRegressor
import pytest
import numpy as np

@pytest.fixture(params=["nearest_n", "noisy", "window", "vanilla", "auto"])
def reg_maker(request):
    match request.param:
        case "nearest_n":
            return NearestNeighborRegressor
        case "noisy":
            return SymmetricNoisyRegressor
        case "window":
            return WindowRegressor
        case "vanilla":
            return VanillaOnlineRegressor
        case "auto":
            return AutoRegressor
        case _:
            raise Exception()

def test_can_run_1d(reg_maker, rng):
    w = np.array([2, -3])
    def f(point):
        return w @ point + 4
    space = np.linspace(0, 1, 100)

    reg = reg_maker(2, 1)
    for _ in range(1_000):
        x = rng.choice(space, size=2)
        y = f(x)
        pred = reg.predict(x)
        if not np.any(np.isnan(pred)):
            assert np.linalg.norm(pred - y) < 1e2
        reg.safe_observe(x=x, y=y)

def test_can_run_nd(reg_maker, rng):
    m, n = 4, 3
    w = rng.random(size=(m, n))
    def f(point):
        return w @ point + 4
    space = np.linspace(0, 1, 100)

    reg = reg_maker(n, m)
    for _ in range(1_000):
        x = rng.choice(space, size=n)
        y = f(x)
        pred = reg.predict(x)
        if not np.any(np.isnan(pred)):
            assert np.linalg.norm(pred - y) < 1e2
        reg.safe_observe(x=x, y=y)

def test_nan_at_right_time(reg_maker, rng):
    m, n = 4, 3
    space = np.linspace(0, 1, 100)

    w = rng.random(size=(m, n))
    def f(point):
        return w @ point + 4

    reg = reg_maker(n, m)
    assert np.all(np.isnan(reg.predict(rng.choice(space, size=n))))
    for _ in range(1_000):
        x = rng.choice(space, size=n)
        y = f(x)
        reg.safe_observe(x=x, y=y)

    x = rng.choice(space, size=n)
    assert np.all(~np.isnan(reg.predict(x)))


def test_output_shapes_are_correct(reg_maker, rng):
    for n, m in [(1,1), (1, 3), (3,1), (3,4)]:
        reg = reg_maker(n, m)
        assert reg.predict(np.zeros(n)).shape == (m,)

        n_samples = 1_000
        inputs = rng.normal(size=(n_samples, n))
        outputs = rng.normal(size=(n_samples, m))
        for i in range(n_samples):
            reg.safe_observe(inputs[i], outputs[i])

        assert reg.predict(np.zeros(n)).shape == (m,)



def test_will_ignore_nan_inputs(reg_maker, rng):
    for n, m in [(1,1), (1, 3), (3,1), (3,4)]:
        reg = reg_maker(n, m)

        n_samples = 1_000
        inputs = rng.normal(size=(n_samples, n))
        outputs = rng.normal(size=(n_samples, m))

        mask = rng.random(size=n_samples) < 0.15
        inputs[mask] *= np.nan

        mask = rng.random(size=n_samples) < 0.15
        outputs[mask] *= np.nan

        for i in range(n_samples):
            reg.safe_observe(inputs[i], outputs[i])

        assert np.all(np.isfinite(reg.predict(np.zeros(n))))

# todo:
#  special auto regressor tests for history dependency