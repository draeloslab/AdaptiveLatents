import adaptive_latents.input_sources.jpca as jpca
import numpy as np

def test_make_H(rng, dd=10):
    for d in range(2,dd):
        H = jpca.make_H(d)
        for _ in range(100):
            k = rng.normal(size=int(d*(d-1)/2))
            sksym = (H @ k).reshape((d,d))
            assert np.linalg.norm(np.real(np.linalg.eigvals(sksym))) < 1e-14




def test_works_on_circular_data():
    assert not np.allclose(U[:,:2], true_variables['C'])
    assert np.allclose(*align_column_spaces(U[:,:2], true_variables['C']))