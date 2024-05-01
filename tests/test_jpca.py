import adaptive_latents.transforms.jpca as jpca
import numpy as np

def test_make_H(rng, dd=10):
    for d in range(2,dd):
        H = jpca.sjPCA.make_H(d)
        for _ in range(100):
            k = rng.normal(size=int(d*(d-1)/2))
            sksym = (H @ k).reshape((d,d))
            assert np.linalg.norm(np.real(np.linalg.eigvals(sksym))) < 1e-14




def test_works_on_circular_data(rng):
    X, X_dot, true_variables = jpca.generate_circle_embedded_in_high_d(rng, m=10_000, stddev=.01)

    X_realigned, U = jpca.sjPCA.apply_to_data(X)
    assert not np.allclose(U[:,:2], true_variables['C'])

    aligned_U, aligned_C = jpca.align_column_spaces(U[:, :2], true_variables['C'])
    assert np.allclose(aligned_U, aligned_C, atol=1e-4)