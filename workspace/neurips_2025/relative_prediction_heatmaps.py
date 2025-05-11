import numpy as np
import matplotlib.pyplot as plt

from adaptive_latents import datasets, proSVD, Pipeline, CenteringTransformer, StreamingKalmanFilter, Bubblewrap, sjPCA, mmICA

from adaptive_latents.predictor import Predictor
from adaptive_latents.regressions import BaseKernelRegressor

if __name__ == '__main__':
    d = datasets.Odoherty21Dataset()
    # pred = MultiPredictor([StreamingKalmanFilter(), Bubblewrap()], log_level=5, check_dt=True)
    prosvd_k = 6

    c = CenteringTransformer()
    svd = proSVD(k=prosvd_k)

    dim_red_methods = [Pipeline(), sjPCA(), mmICA()]
    predictors = [StreamingKalmanFilter() for _ in dim_red_methods]
    regs = [BaseKernelRegressor(maxlen=1000) for _ in dim_red_methods]

    p = Pipeline([c, svd])

    for data in p.streaming_run_on(d.neural_data):

        mse_s = []
        in_space_data = []
        for dim_red_method, predictor, reg in zip(dim_red_methods, predictors, regs):
            in_space_datum = dim_red_method.partial_fit_transform(data)
            in_space_data.append(in_space_datum)

            mse = ((in_space_datum - predictor.predict(1)) ** 2).mean()
            mse_s.append(mse)
            predictor.partial_fit_transform(in_space_datum)

        best_regressor = np.argmin(mse_s)
        for i, (reg, in_space_datum) in enumerate(zip(regs, in_space_data)):
            reg.observe(in_space_datum, np.array([i == best_regressor]))

    fig, axs = plt.subplots(3, 3, figsize=(10, 3), squeeze=False, sharex='row')
    for reg, ax in zip(regs, axs[0]):
        ax.scatter(reg.history[:,0], reg.history[:,1], c=reg.history[:,-1])

    for reg, ax in zip(regs, axs[2]):
        reg:BaseKernelRegressor
        length_scales = np.logspace(-3.5, .5, 20)
        best_scale, (length_scales, errors, error_stds) = reg.cross_validate_length_scale(length_scales, depth=1)
        reg.length_scale = best_scale
        ax.plot(length_scales, errors+error_stds)
        ax.semilogx()

    for reg, ax, old_ax in zip(regs, axs[1], axs[0]):
        reg:BaseKernelRegressor
        Predictor.plot_pdf(fig, ax, reg.predict, xlim=old_ax.get_xlim(), ylim=old_ax.get_ylim(), density=100, native_d=prosvd_k,)



    plt.show(block=True)