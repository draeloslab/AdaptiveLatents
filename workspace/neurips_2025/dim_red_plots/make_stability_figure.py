import matplotlib.pyplot as plt
import numpy as np
from adaptive_latents import ArrayWithTime


def make_stability_figure(Qs):
    fig, ax = plt.subplots()
    dQ = np.linalg.norm(np.diff(Qs, axis=0), axis=1)
    dQ = ArrayWithTime(dQ, Qs.t[1:])
    ax.plot(dQ.t, np.log(dQ))

    return fig
