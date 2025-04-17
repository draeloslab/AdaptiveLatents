import matplotlib

import adaptive_latents
from adaptive_latents import Bubblewrap, ArrayWithTime

import matplotlib.pyplot as plt


class TestBubblewrap:
    def test_plots(self, rng, show_plots):
        if not show_plots:
            matplotlib.use('Agg')

        bw = Bubblewrap(num=10, M=10, log_level=2, check_dt=True)

        hmm = adaptive_latents.input_sources.hmm_simulation.HMM.gaussian_clock_hmm()
        states, observations = hmm.simulate_with_states(n_steps=50, rng=rng)

        bw.offline_run_on(observations)

        fig, axs = plt.subplots(nrows=4, ncols=4)
        axs = axs.flatten()

        i = -1
        bw.show_bubbles_2d(axs[(i := i + 1)])
        bw.show_alpha(axs[(i := i + 1)])
        bw.show_active_bubbles_2d(axs[(i := i + 1)])
        bw.show_active_bubbles_and_connections_2d(axs[(i := i + 1)], observations)
        bw.show_A(axs[(i := i + 1)])
        bw.show_nstep_pdf(ax=axs[(i := i + 1)], other_axis=axs[0], fig=fig, density=2)

        bws = [bw]
        Bubblewrap.compare_runs(bws)

        bws = [bw,bw]
        behavior_dicts = [{'predicted_behavior': ArrayWithTime.from_notime(rng.normal(size=(20,3))), 'true_behavior':ArrayWithTime.from_notime(rng.normal(size=(10,3)))} for _ in range(len(bws))]
        Bubblewrap.compare_runs(bws, behavior_dicts)

        if show_plots:
            plt.show(block=True)
