from adaptive_latents.transforms import proSVD
from adaptive_latents.transforms.jpca import align_column_spaces
import numpy as np

def column_space_distance(Q1, Q2):
    Q1_rotated, Q2 = align_column_spaces(Q1, Q2)
    return np.linalg.norm(Q1_rotated - Q2)

def probabilistically_check_adding_channels_works(rng, n_samples=50, n1=4,n2=10,k=2):
    errors = [[],[]]
    for _ in range(n_samples):
        d = np.ones(n2)
        d[0] = 1.3
        d[-1] = 10
        d = np.diag(d)
        sub_d = d[:-(n2-n1),:-(n2-n1)] if n1 != n2 else d

        data = [
            sub_d @ rng.normal(size=(n1,n1)),
            sub_d @ rng.normal(size=(n1,5)),
            d @ rng.normal(size=(n2,n2)),
            d @ rng.normal(size=(n2,5)),
            ]


        psvd1 = proSVD(k)
        psvd2 = proSVD(k)

        psvd1.initialize(data[2])
        for i in np.arange(data[3].shape[1]):
            psvd1.updateSVD(data[3][:,i:i+1])

        psvd2.initialize(data[0])
        for i in np.arange(data[1].shape[1]):
            psvd2.updateSVD(data[1][:, i:i + 1])

        psvd2.add_new_input_channels(n2-n1)
        for j in [2,3]:
            for i in np.arange(data[j].shape[1]):
                psvd2.updateSVD(data[j][:, i:i + 1])

        ideal_basis = np.zeros((n2,2))
        ideal_basis[0,0] = 1
        ideal_basis[-1, 1] = 1

        errors[0].append(column_space_distance(psvd1.Q, ideal_basis))
        errors[1].append(column_space_distance(psvd2.Q, ideal_basis))
    errors = np.array(errors)
    diff = errors[0] - errors[1]
    return (errors[0] - errors[1] > 0).mean(), diff.mean()

def test_adding_colums_doesnt_hurt(rng):
    assert probabilistically_check_adding_channels_works(rng)[0] > .5