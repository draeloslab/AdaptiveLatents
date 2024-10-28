import adaptive_latents
from adaptive_latents import Bubblewrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TestBubblewrap:
    def test_plots(self, rng):
        bw = Bubblewrap(num=10, M=10, log_level=1)

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
        Bubblewrap.compare_runs([bw])
