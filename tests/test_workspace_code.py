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
        datasets.Churchland10Dataset,
        datasets.Nason20Dataset,
        datasets.Temmar24uDataset,
        datasets.Musall19Dataset,
    ]

    mult_datasets = [
        # datasets.Low21Dataset,
        datasets.Odoherty21Dataset,
        datasets.Schaffer23Datset,
        datasets.Peyrache15Dataset,
        datasets.Naumann24uDataset,
    ]

    @pytest.mark.parametrize("DatasetClass", singleton_datsets)
    def test_can_load_singletons(self, DatasetClass):
        d = DatasetClass()
        assert d.neural_data is not None
        assert d.behavioral_data is not None

    @pytest.mark.parametrize("DatasetClass", mult_datasets)
    def test_can_load_multis(self, DatasetClass):
        for sub_dataset in DatasetClass.sub_datasets:
            d = DatasetClass(sub_dataset_identifier=sub_dataset)
            assert hasattr(d, 'sub_dataset')

            assert d.neural_data is not None
            assert d.behavioral_data is not None
            assert d.opto_stimulations is not None
