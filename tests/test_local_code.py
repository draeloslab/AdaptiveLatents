import pytest
from workspace.main import main
from adaptive_latents import datasets, CenteringTransformer
from adaptive_latents import prediction_regression_run as prr
import matplotlib.pyplot as plt

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
        datasets.Odoherty21Dataset,
        # datasets.Musall19Dataset, # TODO: this takes too long to run
    ]

    mult_datasets = [
        # datasets.Low21Dataset,
        datasets.Zong22Dataset,
        datasets.Schaffer23Datset,
        datasets.Peyrache15Dataset,
        datasets.Naumann24uDataset,
        datasets.TostadoMarcos24Dataset,
    ]


    @pytest.mark.parametrize("DatasetClass", singleton_datsets)
    def test_can_load_singletons(self, DatasetClass):
        d = DatasetClass()
        assert d.neural_data is not None
        assert d.behavioral_data is not None

        self.check_runs_in_pipeline(d)

    @pytest.mark.parametrize("DatasetClass", mult_datasets)
    def test_can_load_multis(self, DatasetClass):
        for sub_dataset in DatasetClass.sub_datasets:
            d = DatasetClass(sub_dataset_identifier=sub_dataset)
            assert hasattr(d, 'sub_dataset')

            assert d.neural_data is not None
            assert d.behavioral_data is not None or d.opto_stimulations is not None

            self.check_runs_in_pipeline(d)

    @staticmethod
    def check_runs_in_pipeline(d):
        iterator = CenteringTransformer().streaming_run_on(d.neural_data)
        for _ in range(10):
            next(iterator)


@longrun
def test_all_prr_defaults():
    prr.pred_reg_run_with_defaults('naumann24u', exit_time=60)
    prr.pred_reg_run_with_defaults('zong22', exit_time=60)
