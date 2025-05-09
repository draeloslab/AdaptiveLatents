import matplotlib.pyplot as plt
from adaptive_latents.input_sources.lds_simulation import LDS
import numpy as np

from common import log_for_tex

def show_toy_dataset():
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

    rng = np.random.default_rng(0)

    def u_function(lds, state, i, rng):
        u = np.zeros(lds.B.shape[0])
        if i == 20:
            u[2] = 5
        return u

    show_toy_n_turns = log_for_tex(key='show_toy_n_turns', value=10, current_file=__file__, output_directory=args.output.parent)

    X, Y, stim = LDS.run_nest_dynamical_system(show_toy_n_turns, radius=10, rng=rng, u_function=u_function)

    ax.plot(Y[:, 0], Y[:, 1], Y[:, 2])
    ax.axis('equal')

    return fig

if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    fig = show_toy_dataset()

    fig.savefig(args.output, bbox_inches="tight")
