from adaptive_latents.stim_optimization import StimDesigner
import numpy as np

def test_design_stim(rng: np.random.Generator):

    for max_nonzero_elements in [1, 5, 10, 100]:
        sd = StimDesigner(max_l0_norm=max_nonzero_elements,)
        v = rng.normal(size=(100,1))
        v = v/np.linalg.norm(v)
        s = sd.design_stim(v, rng=rng)
        assert np.linalg.norm(s, ord=0) <= max_nonzero_elements
