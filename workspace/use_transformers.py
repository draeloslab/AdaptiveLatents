from adaptive_latents.transforms.transformer import CenteringTransformer, Pipeline
from adaptive_latents.transforms.prosvd import TransformerProSVD
from adaptive_latents.transforms.jpca import TransformerSJPCA
from adaptive_latents.transforms.ica import TransformerMMICA
import tqdm

from datasets import Churchland22Dataset
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

if __name__ == '__main__':
    d = Churchland22Dataset()
    pipeline = Pipeline([
        CenteringTransformer(),
        TransformerProSVD(k=10, init_size=20, whiten=True),
        TransformerSJPCA(),
        TransformerMMICA(),
    ])
    pipeline.offline_fit_transform(d.neural_data)

    # x = d.neural_data.a[200:2_000,None,:]
    # output = pipeline.offline_fit_transform(tqdm.tqdm(x))
    #
    # output = np.squeeze(output)
    # plt.matshow(np.cov(output[500:].T))
    # plt.show()

    # plt.scatter(output[50:,9], output[50:,1])
    # plt.axis('equal')
