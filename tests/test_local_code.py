import pytest
from workspace.main import main
import adaptive_latents.datasets as datasets
import workspace.leventhal_script as leventhal_script
import matplotlib.pyplot as plt

longrun = pytest.mark.skipif("not config.getoption('longrun')")


class TestScripts:
    def test_run_main(self, outdir):
        main(output_directory=outdir, steps_to_run=35)

    @longrun
    def test_leventhal_script(self, outdir):
        leventhal_script.make_video(outdir=outdir)
        leventhal_script.show_events_timestamps_and_average_trace(show=False)
        leventhal_script.show_response_arcs(show=False)



@longrun
class TestDatasets:
    singleton_datsets = [
        datasets.Churchland10Dataset,
        datasets.Nason20Dataset,
        datasets.Temmar24uDataset,
        # datasets.Musall19Dataset, # TODO: this takes too long to run
    ]

    mult_datasets = [
        # datasets.Low21Dataset,
        datasets.Odoherty21Dataset,
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

    @pytest.mark.parametrize("DatasetClass", mult_datasets)
    def test_can_load_multis(self, DatasetClass):
        for sub_dataset in DatasetClass.sub_datasets:
            d = DatasetClass(sub_dataset_identifier=sub_dataset)
            assert hasattr(d, 'sub_dataset')

            assert d.neural_data is not None
            assert d.behavioral_data is not None or d.opto_stimulations is not None

@longrun
def test_dataset_plots():
    d = datasets.TostadoMarcos24Dataset()
    fig, ax = plt.subplots()
    d.play_audio()
    d.plot_recalculated_spectrogram(ax)