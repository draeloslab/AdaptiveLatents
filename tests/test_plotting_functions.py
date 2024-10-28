import adaptive_latents.plotting_functions as bpf


def test_makes_movie(tmp_path):
    with bpf.AnimationManager(outdir=tmp_path) as am:
        for i in range(2):
            for ax in am.axs.flatten():
                ax.cla()
            # animation things would go here
            am.grab_frame()
        fpath = am.outfile
    assert fpath.is_file()
