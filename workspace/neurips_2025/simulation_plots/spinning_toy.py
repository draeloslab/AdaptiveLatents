from learn_s_hat_toy import make_srs
import numpy as np
import matplotlib.pyplot as plt
import copy

if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    srs1 = make_srs(copy.deepcopy(rng), n_runs=1, u_function='curvy', show_tqdm=True)
    srs2 = make_srs(copy.deepcopy(rng), n_runs=1, u_function='curvy spins', show_tqdm=True)

    fig, ax = plt.subplots()

    err = srs2['learning from stim'][0].log['pred_error'][:,2]
    ax.plot(err.t, err, label='with spin')

    err = srs1['learning from stim'][0].log['pred_error'][:,2]
    ax.plot(err.t, err, label='no spin')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('error')

    fig.savefig(args.output)