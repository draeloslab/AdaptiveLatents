from adaptive_latents.transforms.transformer import CenteringTransformer, Pipeline
from adaptive_latents.transforms.proSVD import TransformerProSVD
from adaptive_latents.transforms.jpca import TransformerSJPCA
from adaptive_latents.transforms.ica import TransformerMMICA
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

if __name__ == '__main__':
    pipeline = Pipeline([
        CenteringTransformer(),
        TransformerProSVD(k=10, init_size=20, whiten=True),
        TransformerSJPCA(),
        TransformerMMICA(),
    ])

    x = rng.normal(size=(500,1,10)) * np.arange(10) + 500
    output = pipeline.offline_fit_transform(x)

    output = np.squeeze(output)
    assert np.all(~np.isnan(output[100:]))
    plt.matshow(np.cov(output[100:].T))

    # plt.scatter(output[50:,9], output[50:,1])
    # plt.axis('equal')
    plt.show()
