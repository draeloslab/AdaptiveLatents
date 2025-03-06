import pytest
import numpy as np
import inspect


# https://stackoverflow.com/a/43938191
def pytest_addoption(parser):
    parser.addoption('--longrun', action='store_true', dest="longrun", default=False, help="enable 'longrun' decorated tests")
    parser.addoption('--showplots', action='store_true', dest="showplots", default=False, help="show plots from matplotlib tests")


@pytest.fixture
def show_plots(pytestconfig):
    return pytestconfig.getoption('showplots')


@pytest.fixture
def outdir(tmpdir):
    tmpdir.mkdir("generated")
    outdir = tmpdir.mkdir("generated/bubblewrap_runs")
    return outdir


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def get_all_subclasses(cls, concrete=True):
    unchecked_subclasses = list(cls.__subclasses__())
    subclasses = []
    while unchecked_subclasses:
        cls = unchecked_subclasses.pop()
        unchecked_subclasses = unchecked_subclasses + list(set(cls.__subclasses__()) - set(subclasses))
        subclasses.append(cls)
    if concrete:
        subclasses = [subclass for subclass in subclasses if not inspect.isabstract(subclass)]
    return subclasses

