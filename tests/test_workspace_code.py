import pytest
from workspace.main import main
import workspace.datasets as datasets

longrun = pytest.mark.skipif("not config.getoption('longrun')")

class TestScripts:
    def test_run_main(self, outdir):
        main(output_directory=outdir, steps_to_run=35)


@longrun
class TestDatasets:
    singleton_datsets = [
        datasets.Churchland22Dataset,
        datasets.Nason20Dataset,
        datasets.Temmar24uDataset,
        datasets.Musall19Dataset,
    ]

    mult_datasets = [
        # datasets.Low21Dataset,
        datasets.Odoherty21Dataset,
        datasets.Schaffer23Datset,
        datasets.Peyrache15Dataset,
    ]

    @pytest.mark.parametrize("dataset_class", singleton_datsets)
    def test_can_load_singletons(self, dataset_class):
        d = dataset_class()
        d.construct()


    @pytest.mark.parametrize("dataset_class", mult_datasets)
    def test_can_load_multis(self, dataset_class):
        d = dataset_class()
        for sub_dataset in d.get_sub_datasets():
            d.construct(sub_dataset_identifier=sub_dataset)
