import numpy as np
import matplotlib.pyplot as plt
import adaptive_latents as bw
import adaptive_latents.input_sources.functional as fin
import tqdm as tqdm
from adaptive_latents.input_sources.data_sources import NumpyTimedDataSource

def main():
    ca, vid, t_ca, t_vid = fin.generate_musal_dataset(cam=1, video_target_dim=100, resize_factor=1, prosvd_init_size=100)

    pre_datasets = {
        'vid # i': (vid, t_vid),
        'ca # i': (ca, t_ca),
        's(ca,1) # o': fin.clip(fin.prosvd_data(ca, 1, 20), t_ca),
        's(vid,1) # o': fin.clip(fin.prosvd_data(vid, 1, 20), t_ca)
    }

    datasets = {}
    input_keys = []
    output_keys = []
    for key, value in pre_datasets.items():
        k, tags = key.split("#")
        k = k.strip()
        datasets[k] = value
        if "i" in tags:
            input_keys.append(k)
        if "o" in tags:
            output_keys.append(k)


    def evaluate(i,o, maxlen=30_000):
        i, i_t, o, o_t = fin.clip(*i, *o, maxlen=maxlen)

        o_dt = np.median(np.diff(o_t))
        i_dt = np.median(np.diff(i_t))
        n_steps = int(np.ceil(o_dt/i_dt))

        br = bw.bw_run.simple_bw_run(input_arr=i,t=i_t, time_offsets=[0,n_steps], bw_params= bw.default_parameters.default_jpca_dataset_parameters)

        alpha_dict = br.alpha_history
        a_current, a_ahead, o, o_t = fin.clip(alpha_dict[0], alpha_dict[1], o, o_t)
        reg = bw.SymmetricNoisyRegressor(input_d=a_current.shape[1], output_d=o.shape[1], init_min_ratio=5)

        pred, true, times = br.evaluate_regressor(reg, o, o_t, test_offset=n_steps)
        return br, pred, true

    results = {}
    brs = {}
    true_values = {}
    for okey in output_keys:
        results[okey] = {}
        brs[okey] = {}
        for ikey in input_keys:
            print(f"{okey= } {ikey= }")

            br, pred, true = evaluate(datasets[ikey], datasets[okey], maxlen=1_000)

            results[okey][ikey] = pred
            brs[okey][ikey] = br

            if okey not in true_values:
                true_values[okey] = true


if __name__ == '__main__':
    main()