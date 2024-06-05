import adaptive_latents as al
def make_data():
    import adaptive_latents.transforms.utils as fin
    import adaptive_latents
    import numpy as np

    def smooth_columns(X, t, kernel_length=5, kernel_end=-3):
        kernel = np.exp(np.linspace(0, kernel_end, kernel_length))
        kernel /= kernel.sum()
        mode = 'valid'
        X = np.column_stack([np.convolve(kernel, column, mode) for column in X.T])
        t = np.convolve(np.hstack([[1], kernel[:-1] * 0]), t, mode)
        return X, t


    obs, beh, obs_t, beh_t = adaptive_latents.input_sources.datasets.construct_jenkins_data(bin_width=0.06)
    beh, beh_t = fin.resample_matched_timeseries(beh, obs_t, beh_t), obs_t
    obs, obs_t = smooth_columns(obs, obs_t, kernel_length=20)

    return fin.clip(obs, beh)


bw_params = dict(
    al.default_parameters.default_jpca_dataset_parameters,
    M = 100,
    num=100,
    eps=1e-3,
    step=8e-1,
    num_grad_q=2,
)
del bw_params['go_fast']

if __name__ == '__main__':
    A, beh = make_data()

    t = al.profiling_functions.get_speed_by_time(A, beh, prosvd_k=4, bw_params=bw_params, max_steps=5000)

    import numpy as np
    import matplotlib.pyplot as plt
    t *= 1000

    print(np.median(np.diff(t)))
    plt.plot((t[:-1] - t[0]), np.diff(t), '-')
    plt.show()
