from adaptive_latents.transforms import proSVD
from adaptive_latents.transforms.jpca import align_column_spaces
import numpy as np

def compare_methods(rng):
    d = np.ones(8)
    d[0] = 1.01
    d[-1] = 1.1
    d = np.diag(d)

    data = [
        d[:-3,:-3] @ rng.normal(size=(5,50)),
        d @ rng.normal(size=(8,8)),
        d @ rng.normal(size=(8,100)),
        ]


    psvd1 = proSVD(2)
    psvd2 = proSVD(2)

    psvd1.initialize(data[1])
    for i in np.arange(data[2].shape[1]):
        psvd1.updateSVD(data[2][:,i:i+1])

    psvd2.initialize(data[0])
    psvd2.add_new_input_channels(3)
    for j in [1,2]:
        for i in np.arange(data[j].shape[1]):
            psvd2.updateSVD(data[j][:, i:i + 1])

    ideal_basis = np.zeros((8,2))
    ideal_basis[0,0] = 1
    ideal_basis[-1, 1] = 1

    Q1, _ = align_column_spaces(psvd1.Q, ideal_basis)
    Q2, _ = align_column_spaces(psvd2.Q, ideal_basis)
    return ((Q1 - ideal_basis)**2).sum() > ((Q2 - ideal_basis)**2).sum()

def test_can_add_columns(rng):
    rate = np.mean([compare_methods(rng) for _ in range(200)])
    assert  rate > .5