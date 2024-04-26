from adaptive_latents.regressions import NearestNeighborRegressor, SymmetricNoisyRegressor, WindowRegressor, VanillaOnlineRegressor, auto_regression_decorator
import pytest
import numpy as np

@pytest.fixture(params=["nearest_n", "noisy", "window", "vanilla"])
def base_reg_maker(request):
    match request.param:
        case "nearest_n":
            return NearestNeighborRegressor
        case "noisy":
            return SymmetricNoisyRegressor
        case "window":
            return WindowRegressor
        case "vanilla":
            return VanillaOnlineRegressor
        case _:
            raise Exception()

@pytest.fixture(params=["no autoregression", "autoregression 0", "autoregression 2"])
def reg_maker(request, base_reg_maker):
    match request.param:
        case "no autoregression":
            return base_reg_maker
        case "autoregression 0":
            return auto_regression_decorator(base_reg_maker, n_steps=0)
        case "autoregression 2":
            return auto_regression_decorator(base_reg_maker, n_steps=2)
        # case "autoregression only":
        #     return auto_regression_decorator(base_reg_maker, history_only=True)
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
        reg.observe(x=x, y=y)

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
        reg.observe(x=x, y=y)

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
        reg.observe(x=x, y=y)

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
            reg.observe(inputs[i], outputs[i])

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
            reg.observe(inputs[i], outputs[i])

        # make a clear history if we're testing an autoregressor
        if hasattr(reg, "_y_history"):
            for _ in range(reg._y_history.maxlen):
                reg.observe(rng.normal(size=n), rng.normal(size=m))

        assert np.all(np.isfinite(reg.predict(np.zeros(n))))

# todo:
#  special auto regressor tests for history dependency