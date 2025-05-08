import numpy as np
import pytest

import matplotlib.pyplot as plt
import copy

from adaptive_latents.regressions import BaseKNearestNeighborRegressor, BaseVanillaOnlineRegressor, VanillaOnlineRegressor, BaseKernelRegressor, auto_regression_decorator


longrun = pytest.mark.skipif("not config.getoption('longrun')")



@pytest.fixture(params=["nearest_n", "kernel", "vanilla", "vanilla_regularized"])
def base_reg_maker(request):
    if request.param == "nearest_n":
        return BaseKNearestNeighborRegressor
    if request.param == "kernel":
        return BaseKernelRegressor
    elif request.param == "vanilla":
        class NoRegularizationTempVersion(BaseVanillaOnlineRegressor):
            def __init__(self, *args, regularization_factor=0, **kwargs):
                super().__init__(*args, regularization_factor=regularization_factor, **kwargs)
        return NoRegularizationTempVersion
    elif request.param == "vanilla_regularized":
        return BaseVanillaOnlineRegressor
    raise NotImplementedError()


@pytest.fixture(params=["no autoregression", "autoregression 0", "autoregression 2"])
def reg_maker(request, base_reg_maker):
    if request.param == "no autoregression":
        return base_reg_maker
    elif request.param == "autoregression 0":
        return auto_regression_decorator(base_reg_maker, n_steps=0)
    elif request.param == "autoregression 2":
        return auto_regression_decorator(base_reg_maker, n_steps=2)
    raise NotImplementedError()


def test_can_run_nd(reg_maker, rng):
    m, n = 4, 3
    w = rng.random(size=(m, n))

    def f(point):
        return w@point + 4

    space = np.linspace(0, 1, 100)

    reg = reg_maker()
    for i in range(50):
        x = rng.choice(space, size=n)
        y = f(x)
        pred = reg.predict(x)
        if i > 10 and not np.any(np.isnan(pred)):
            assert np.linalg.norm(pred - y) < 1e2
        reg.observe(x=x, y=y)


def test_output_shapes_are_correct(reg_maker, rng):
    for n, m in [(1, 1), (1, 3), (3, 1), (3, 4)]:
        reg = reg_maker()

        n_samples = 10
        inputs = rng.normal(size=(n_samples, n))
        outputs = rng.normal(size=(n_samples, m))
        for i in range(n_samples):
            reg.observe(inputs[i], outputs[i])

        assert reg.predict(np.zeros(n)).shape == (m,)


def test_will_ignore_nan_inputs(reg_maker, rng):
    for n, m in [(1, 1), (1, 3), (3, 1), (3, 4)]:
        reg = reg_maker()

        n_samples = 50
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

@longrun
def test_cross_validate_length_scale(rng, show_plots):
    rng = np.random.default_rng(3)
    def f(x, space_constant=1):
        return np.cos(x*space_constant).mean()

    outputs = []

    x = rng.uniform(-2 * np.pi, 2 * np.pi, size=(500,2))
    noise = rng.normal(size=x.shape[0], scale=.5)

    space_constants = np.logspace(-1,1,5)
    for space_constant in space_constants:
        reg = BaseKernelRegressor()
        for x_sample, noise_sample in zip(x, noise):
            x_sample = x_sample / space_constant
            y = f(x_sample, space_constant=space_constant) + noise_sample
            reg.observe(x_sample, y)

        length_scales = np.logspace(-4, 4, 17)
        output = reg.cross_validate_length_scale(length_scales, depth=50, ratio=.9, rng=copy.deepcopy(rng))
        outputs.append(output)

    if show_plots:
        fig, ax = plt.subplots()
        for output in outputs:
            length_scales, errors, error_stds = output[1]
            ax.plot(length_scales, errors + error_stds)
        ax.semilogx()
        plt.show(block=True)

    found_constants = np.log([o[0] for o in outputs])
    diffs = np.diff(found_constants)
    assert np.allclose(diffs, diffs[0], atol=.001)




# todo:
#  special auto regressor tests for history dependency
