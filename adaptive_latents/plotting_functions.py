import datetime
import pathlib
import matplotlib.pylab as plt
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegFileWriter
from adaptive_latents import CONFIG
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adaptive_latents import CONFIG


class AnimationManager:
    def __init__(self, filename=None, outdir=None, n_rows=1, n_cols=1, fps=20, dpi=100, extension="mp4", figsize=(10,10), projection='rectilinear', make_axs=True):
        outdir = outdir or CONFIG['plot_save_path']
        if filename is None:
            time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            filename = f'movie_{time_string}'
        self.outfile = pathlib.Path(outdir).resolve() / f"{filename}.{extension}"
        self.movie_writer = FFMpegFileWriter(fps=fps)
        if make_axs:
            # TODO: rename ax to axs
            self.fig, self.axs = plt.subplots(n_rows, n_cols, figsize=figsize, layout='tight', squeeze=False, subplot_kw={'projection': projection})
        else:
            self.fig = plt.figure(figsize=figsize, layout='tight')
        self.movie_writer.setup(self.fig, self.outfile, dpi=dpi)
        self.seen_frames = 0
        self.finished = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seen_frames:
            self.finish()

    def finish(self):
        if not self.finished:
            self.movie_writer.finish()
            self.finished = True

    def grab_frame(self):
        self.movie_writer.grab_frame()
        self.seen_frames += 1


def use_bigger_lims(ax, old_lims=None, y=True, x=True):
    new_lims = ax.axis()
    if old_lims is None:
        old_lims = new_lims

    future_lims = [min(old_lims[0], new_lims[0]), max(old_lims[1], new_lims[1]), min(old_lims[2], new_lims[2]), max(old_lims[3], new_lims[3])]
    if not y:
        future_lims[2:] = new_lims[2:]

    if not x:
        future_lims[:2] = new_lims[:2]
    ax.axis(future_lims)
