import adaptive_latents.plotting_functions as pf
import adaptive_latents as al
import numpy as np
import matplotlib.pyplot as plt
import pytest

SHOW = True
@pytest.fixture
def spiral():
    start = np.array([0,0.01])
    theta = 0.1
    A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) + np.diag([theta/10, theta/10])
    X = [start]
    for _ in range(400):
        X.append(A @ X[-1])
    return al.ArrayWithTime.from_notime(X)


def test_makes_movie(tmp_path):
    with pf.AnimationManager(outdir=tmp_path) as am:
        for i in range(2):
            for ax in am.axs.flatten():
                ax.cla()
            # animation things would go here
            am.grab_frame()
        fpath = am.outfile
    assert fpath.is_file()


def test_plot_flow_fields(spiral):
    pf.plot_flow_fields({'spiral1': spiral, 'spiral2':spiral}, normalize_method='squares', grid_n=20)
    if SHOW:
        plt.show()

def test_updating_plot():
    import time
    o = pf.UpdatingOptimizationGraph()
    for v in o.suggest_values(0, 6.28):
        # time.sleep(.01)
        o.register_result(v, {'beh': {'corr': [np.sin(v), np.cos(v)], 'nrmse': -np.sin(v)},
                               'joint': {'corr': [np.cos(v), np.cos(v * 2)], 'nrmse': np.cos(v)}})

        if len(o.tried_values) > 20:
            break

    if SHOW:
        plt.show()

def test_use_bigger_lims():
    fig1, ax1 = plt.subplots()
    ax1.plot([0, 1], [0, 1])

    fig2, ax2 = plt.subplots()
    ax2.plot([0, .1], [0, .1])
    pf.use_bigger_lims(ax2, old_lims=ax1.axis())
    if SHOW:
        plt.show()

def test_history_with_tail(spiral):
    fig, ax = plt.subplots()
    pf.plot_history_with_tail(ax, data=spiral, current_t=200, tail_length=4, scatter_alpha=1)
    if SHOW:
        plt.show()


def test_prediction_video():
    pf.PredictionVideo.example_usage()
    if SHOW:
        plt.show()
