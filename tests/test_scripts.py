import pytest
import scripts
from scripts.main import main
import scripts.datasets as datasets

longrun = pytest.mark.skipif("not config.getoption('longrun')")

class TestScripts:
    def test_run_main(self, outdir):
        main(output_directory=outdir, steps_to_run=35)


@longrun
class TestDatasets:
    def test_can_load_indy(self):
        for identifier in scripts.datasets.individual_identifiers["indy"]:
            obs, raw_behavior, obs_t, beh_t = scripts.datasets.construct_indy_data(individual_identifier=identifier, bin_width=.03)

    def test_can_load_jenkins(self):
        obs, raw_behavior, obs_t, beh_t = scripts.datasets.construct_jenkins_data(bin_width=.03)

    def test_can_load_nason20_dataset(self):
        obs, raw_behavior, obs_t, beh_t = scripts.datasets.construct_nason20_data(bin_width=.03)

    def test_can_load_unpublished24(self):
        obs, raw_behavior, obs_t, beh_t = scripts.datasets.construct_unpublished24_data()

    def test_can_load_buzaki(self):
        for identifier in scripts.datasets.individual_identifiers["buzaki"]:
            obs, raw_behavior, obs_t, beh_t = scripts.datasets.construct_buzaki_data(individual_identifier=identifier,
                                                                                 bin_width=.03,
                                                                                 _recalculate_cache_value=True)

    def test_can_load_fly(self):
        for identifier in scripts.datasets.individual_identifiers["fly"]:
            obs, raw_behavior, obs_t, beh_t = scripts.datasets.construct_fly_data(individual_identifier=identifier)

    def test_can_load_musal(self):
        obs, raw_behavior, obs_t, beh_t = scripts.datasets.generate_musal_dataset(_recalculate_cache_value=True)
