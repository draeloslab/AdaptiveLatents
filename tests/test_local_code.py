import functools
import inspect
import pathlib
import sys
import typing

import pytest
from conftest import get_all_subclasses

from adaptive_latents import CenteringTransformer, datasets
from adaptive_latents import prediction_regression_run as prr
from workspace.main import main

longrun = pytest.mark.skipif("not config.getoption('longrun')")


def test_run_main(outdir):
    main(output_directory=outdir, steps_to_run=35)


scripts = sorted(list(pathlib.Path(__file__, '..', '..', 'workspace', 'demos').resolve().glob('demo*.py')))
@longrun
@pytest.mark.parametrize('script', scripts)
def test_script_execution(script, show_plots):
    # there has to be a better way, but lots of things either mess up pickling, coverage, or jax
    sys.path.append(str(script.parent))
    demo = __import__(script.stem)
    main: typing.Callable = demo.main

    kwargs = {}
    if 'show_plots' in inspect.signature(main).parameters:
        kwargs['show_plots'] = show_plots

    main(**kwargs)


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
@pytest.mark.parametrize("dataset,exit_time", [('odoherty21', 60), ('naumann24u',300), ('zong22', 60)])
def test_all_prr_defaults(dataset, exit_time):
    prr.pred_reg_run_with_defaults(dataset, exit_time=exit_time)
