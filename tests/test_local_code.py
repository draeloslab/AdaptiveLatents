import pytest
from workspace.main import main
from adaptive_latents import datasets, CenteringTransformer
from adaptive_latents import prediction_regression_run as prr
from conftest import get_all_subclasses
import functools
import matplotlib.pyplot as plt

longrun = pytest.mark.skipif("not config.getoption('longrun')")


class TestScripts:
    def test_run_main(self, outdir):
        main(output_directory=outdir, steps_to_run=35)


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
    prr.pred_reg_run_with_defaults('naumann24u', exit_time=60)
    prr.pred_reg_run_with_defaults('zong22', exit_time=60)
