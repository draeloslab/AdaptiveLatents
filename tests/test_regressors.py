from adaptive_latents.regressions import NearestNeighborRegressor,  VanillaOnlineRegressor, auto_regression_decorator, SemiRegularizedRegressor
import pytest
import numpy as np


@pytest.fixture(params=["nearest_n",  "vanilla", "vanilla_regularized"])
def base_reg_maker(request):
    match request.param:
        case "nearest_n":
            return NearestNeighborRegressor
        case "vanilla":
            return VanillaOnlineRegressor
        case "vanilla_regularized":
            return SemiRegularizedRegressor
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


def test_can_run_nd(reg_maker, rng):
    m, n = 4, 3
    w = rng.random(size=(m, n))
    def f(point):
        return w @ point + 4
    space = np.linspace(0, 1, 100)

    reg = reg_maker(n, m)
    for i in range(1_000):
        x = rng.choice(space, size=n)
        y = f(x)
        pred = reg.predict(x)
        if i > 10 and not np.any(np.isnan(pred)):
            assert np.linalg.norm(pred - y) < 1e2
        reg.observe(x=x, y=y)


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