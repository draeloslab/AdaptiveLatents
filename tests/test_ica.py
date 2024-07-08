import numpy as np
from adaptive_latents.transforms.ica import mmICA

def test_ica_runs():
    rng = np.random.default_rng()
    ica = mmICA(p=3)
    for _ in range(50):
        data = rng.laplace(size=(10, 10))
        ica.observe_new_batch(data)
    