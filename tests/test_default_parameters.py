import inspect
import adaptive_latents.default_parameters as dp
from adaptive_latents import Bubblewrap
import adaptive_latents as al

# sometimes we add arguments to the Bubblewrap class; this file ensures those arguments are tracked downstream

def test_if_defaults_cover_all_options():
    signature = inspect.signature(Bubblewrap)
    params_with_defaults = {k:v for k, v in signature.parameters.items() if v.default is not signature.empty}
    for v in [v for k,v in dp.__dict__.items() if type(v) == dict and "__" not in k]:
        k1, k2 = set(v.keys()), set(params_with_defaults.keys())

        # testing two differences makes it easier to find where the discrepancy is
        assert k1.difference(k2) == set()
        assert k2.difference(k1) == set()

def test_if_parameter_extraction_misses_none(premade_unfrozen_br):
    bw = premade_unfrozen_br.bw
    params = al.plotting_functions._deduce_bw_parameters(bw)

    signature = inspect.signature(Bubblewrap)
    param_set = {k for k, v in signature.parameters.items()}

    assert param_set.difference(params) == set()