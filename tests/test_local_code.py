import functools
import inspect
import pathlib
import pickle
import runpy
import sys

import pytest
from conftest import get_all_subclasses

from adaptive_latents import CenteringTransformer, datasets
from adaptive_latents import prediction_regression_run as prr
from workspace.main import main

longrun = pytest.mark.skipif("not config.getoption('longrun')")


def test_run_main(outdir):
    main(output_directory=outdir, steps_to_run=35)


scripts = sorted(list(pathlib.Path(__file__, '..', '..', 'workspace', 'demos').resolve().glob('*.py')))
@longrun
@pytest.mark.parametrize('script', scripts)
def test_script_execution(script):
    # this just tests that the scripts run
    # runpy would be better, but it messes with JAX
    # I also considered subprocess, but then we don't get coverage
    # https://stackoverflow.com/a/67692

    main = runpy.run_path(script)['main']
    kwargs = {}
    if 'show' in inspect.signature(main).parameters:
        kwargs['show'] = False

    try:
        main(**kwargs)
    except pickle.PicklingError:
        # Pickle needs to be able to re-find and import the classes in PATH to unpickle them
        assert script.stem == "demo_07"
        sys.path.append(str(script.parent))
        import demo_07
        demo_07.main(**kwargs)



# collect all the datasets
to_test = []
for dataset in get_all_subclasses(datasets.Dataset):
    if hasattr(dataset, 'sub_datasets'):
        for sub_dataset in dataset.sub_datasets:
            to_test.append(functools.partial(dataset, sub_dataset_identifier=sub_dataset))
    else:
        to_test.append(dataset)

@longrun
@pytest.mark.parametrize("dataset", to_test)
def test_all_datasets(dataset):
    d = dataset()
    assert d.neural_data is not None

    iterator = CenteringTransformer().streaming_run_on(d.neural_data)
    for _ in range(10):
        next(iterator)


@longrun
def test_all_prr_defaults():
    prr.pred_reg_run_with_defaults('odoherty21', exit_time=60)
    prr.pred_reg_run_with_defaults('naumann24u', exit_time=300)
    prr.pred_reg_run_with_defaults('zong22', exit_time=60)
