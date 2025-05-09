import matplotlib.pyplot as plt
from adaptive_latents.utils import column_space_distance


def make_nearness_to_offline_figure(Qs, offline_Q):
    fig, ax = plt.subplots()
    differences = [
        column_space_distance(q, offline_Q, method="aligned_diff", override_ortho_check=True)
        for q in Qs
    ]
    ax.plot(Qs.t, differences)
    ax.set_ylim(0, 10)
    return fig
