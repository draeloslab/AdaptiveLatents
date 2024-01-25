from adaptive_latents import Bubblewrap
import datetime
import pickle
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegFileWriter
from .input_sources.functional import save_to_cache
from .input_sources.data_sources import NumpyTimedDataSource
import warnings
import time
from .config import CONFIG

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .regressions import OnlineRegressor

@save_to_cache("simple_bw_run")
def simple_bw_run(input_arr, t, time_offsets, bw_params):
    bw = Bubblewrap(input_arr.shape[1], **bw_params)
    br = BWRun(bw, NumpyTimedDataSource(input_arr, t, time_offsets), show_tqdm=True)
    br.run(save=True)
    return br

class BWRun:
    def __init__(self, bw, obs_ds, beh_ds=None, behavior_regressor=None, animation_manager=None, save_A=False, show_tqdm=True,
                 output_directory=CONFIG["output_path"]/"bubblewrap_runs"):

        self.bw: Bubblewrap = bw
        self.animation_manager: AnimationManager = animation_manager
        self.obs_ds:NumpyTimedDataSource = obs_ds
        self.beh_ds = beh_ds

        if self.obs_ds.output_shape > 10:
            warnings.warn("Bubblewrap might not run well on high-D inputs. Consider using proSVD.")

        # only keep a behavior regressor if there is behavior
        self.behavior_regressor = None
        if self.beh_ds and self.beh_ds.output_shape > 0:
            self.behavior_regressor: OnlineRegressor = behavior_regressor


        time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.output_prefix = os.path.join(output_directory, f"bubblewrap_run_{time_string}")
        self.pickle_file = f"{self.output_prefix}.pickle"
        if self.animation_manager:
            self.animation_manager.set_final_output_location(f"{self.output_prefix}.{self.animation_manager.extension}")

        # self.total_runtime = None
        self.save_A = save_A
        self.show_tqdm = show_tqdm

        # todo: history_of object?
        self.prediction_history = {k: [] for k in obs_ds.time_offsets}
        self.entropy_history = {k: [] for k in obs_ds.time_offsets}
        self.behavior_pred_history = {k: [] for k in obs_ds.time_offsets}
        self.behavior_error_history = {k: [] for k in obs_ds.time_offsets}
        self.alpha_history = {k: [] for k in obs_ds.time_offsets}
        self.bw_timepoint_history = []
        self.reg_timepoint_history = []
        self.runtime = None
        self.runtime_since_init = None

        self.n_living_history = []
        if save_A:
            self.A_history = []

        self.saved = False
        self.frozen = False

        obs_dim = self.obs_ds.output_shape
        assert obs_dim == self.bw.d
        if self.behavior_regressor:
            assert self.beh_ds.output_shape == self.behavior_regressor.output_d
        # note that if there is no behavior, the behavior dimensions will be zero

    def run(self, save=False, limit=None, freeze=True):
        start_time = time.time()
        time_since_init = None

        if len(self.obs_ds) < self.bw.M:
            warnings.warn("Data length shorter than initialization.")

        if limit is None:
            limit = len(self.obs_ds)
        limit = min(len(self.obs_ds), limit)

        bw_step = 0
        obs_next_t, obs_done = self.obs_ds.preview_next_timepoint()
        if self.beh_ds:
            beh_next_t, beh_done = self.beh_ds.preview_next_timepoint()
        else:
            beh_next_t, beh_done = float("inf"), True

        with tqdm(total=limit) as pbar:
            while not (obs_done and beh_done) and bw_step <= limit:
                if beh_done or obs_next_t < beh_next_t:
                    obs = next(self.obs_ds)
                    self.bw.observe(obs)

                    if bw_step < self.bw.M:
                        bw_step += 1
                        pbar.update(1)
                        continue
                    elif bw_step == self.bw.M:
                        self.bw.init_nodes()
                        self.bw.e_step()  # todo: is this OK?
                        self.bw.grad_Q()
                        time_since_init = time.time()
                    else:
                        self.bw.e_step()
                        self.bw.grad_Q()

                    pairs = {}
                    for offset in self.obs_ds.time_offsets:
                        pairs[offset] = self.obs_ds.get_atemporal_data_point(offset)
                    self.bw_log_for_step(bw_step, pairs)
                    bw_step += 1
                    pbar.update(1)
                else:
                    beh = next(self.beh_ds)
                    if hasattr(self.bw, 'alpha'):
                        self.reg_timepoint_history.append(self.beh_ds.current_timepoint())
                        if self.behavior_regressor:
                            self.behavior_regressor.safe_observe(self.bw.alpha, beh)

                            for offset in self.beh_ds.time_offsets:
                                b = self.beh_ds.get_atemporal_data_point(offset)
                                alpha_ahead = self.alpha_history[offset][-1]
                                bp = self.behavior_regressor.predict(alpha_ahead)

                                self.behavior_pred_history[offset].append(bp)
                                self.behavior_error_history[offset].append(bp - b)

                obs_next_t, obs_done = self.obs_ds.preview_next_timepoint()
                if self.beh_ds:
                    beh_next_t, beh_done = self.beh_ds.preview_next_timepoint()
                else:
                    beh_next_t, beh_done = float("inf"), True


        end_time = time.time()
        self.runtime_since_init = end_time - time_since_init
        self.runtime = end_time - start_time

        if freeze:
            self.finish_and_remove_jax()
        if save:
            self.saved = True
            with open(self.pickle_file, "wb") as fhan:
                pickle.dump(self, fhan)

    def bw_log_for_step(self, step, offset_pairs):
        # TODO: allow skipping of (e.g. entropy) steps?
        for offset, o in offset_pairs.items():
            p = self.bw.pred_ahead(self.bw.logB_jax(o, self.bw.mu, self.bw.L, self.bw.L_diag), self.bw.A, self.bw.alpha,
                                   offset)
            self.prediction_history[offset].append(p)

            e = self.bw.get_entropy(self.bw.A, self.bw.alpha, offset)
            self.entropy_history[offset].append(e)
            self.alpha_history[offset].append(self.bw.alpha @ np.linalg.matrix_power(self.bw.A, offset))

            # if self.behavior_regressors:
            #     for i in range(len(self.behavior_regressors)):
            #         alpha_ahead = np.array(self.bw.alpha @ np.linalg.matrix_power(self.bw.A, offset)).reshape(-1, 1)
            #         bp = self.behavior_regressors[i].predict(alpha_ahead)
            #
            #         self.behavior_pred_history[i][offset].append(bp)
            #         self.behavior_error_history[i][offset].append(bp - b)

        self.n_living_history.append(self.bw.N - len(self.bw.dead_nodes))
        if self.save_A:
            self.A_history.append(self.bw.A)

        if self.animation_manager and self.animation_manager.frame_draw_condition(step, self.bw):
            self.animation_manager.draw_frame(step, self.bw, self)

        self.bw_timepoint_history.append(self.obs_ds.current_timepoint())

    def finish_and_remove_jax(self):
        self.frozen = True
        if self.animation_manager:
            self.animation_manager.finish()
            del self.animation_manager

        def convert_dict(d):
            return {k: np.array(v) for k, v in d.items()}

        self.prediction_history = convert_dict(self.prediction_history)
        self.entropy_history = convert_dict(self.entropy_history)
        self.behavior_pred_history = convert_dict(self.behavior_pred_history)
        self.behavior_error_history = convert_dict(self.behavior_error_history)
        self.alpha_history = convert_dict(self.alpha_history)

        self.bw_timepoint_history = np.array(self.bw_timepoint_history)
        self.reg_timepoint_history = np.array(self.reg_timepoint_history)

        self.n_living_history = np.array(self.n_living_history)
        if self.save_A:
            self.A_history = np.array(self.A_history)

        self.bw.freeze()

    # Metrics
    def evaluate_regressor(self, reg, o, o_t, train_offset=0, test_offset=1):
        assert len(o_t) == len(o)
        train_alphas = self.alpha_history[train_offset]
        test_alphas = self.alpha_history[test_offset]
        alpha_t = self.bw_timepoint_history
        pred = []
        truth = []
        pred_times = []

        j = 0
        for i, obs_time in enumerate(o_t):
            while (j+1) < len(alpha_t) and alpha_t[j+1] <= obs_time:
                j += 1
            if j < len(alpha_t) and obs_time <= alpha_t[j]:
                reg.safe_observe(train_alphas[j],o[i])
                pred.append(reg.predict(test_alphas[j]))
                truth.append(o[i])
                pred_times.append(alpha_t[j])
                j += 1

        pred = np.array(pred)
        pred_times = np.array(pred_times)
        truth = np.array(truth)
        if len(pred.shape) == 1:
            pred = pred.reshape(-1, 1)

        return pred, truth, pred_times # predicted, true, times

    def add_regressor_post_hoc(self, reg, o, o_t, train_offset=0, test_offset=1):
        predictions, truth, pred_times = self.evaluate_regressor(reg, o, o_t, train_offset, test_offset)
        self.reg_timepoint_history = np.array(pred_times)
        self.behavior_pred_history = {test_offset: np.array(predictions)}
        self.behavior_error_history = {test_offset: np.array(predictions - truth)}
        return predictions, truth, pred_times # predicted, true, times

    def _last_half_index(self):

        assert self.frozen
        assert self.obs_ds.time_offsets
        pred = self.prediction_history[self.obs_ds.time_offsets[0]]
        assert np.all(np.isfinite(pred))
        return len(pred)//2

    def behavior_pred_corr(self, offset):
        pred, true, err = self.get_behavior_last_half(offset)
        assert np.all(np.isfinite(true))
        return np.corrcoef(pred, true)[0,1]

    def get_behavior_last_half(self, offset):
        i = self._last_half_index()
        pred = self.behavior_pred_history[offset][-i:]
        err = self.behavior_error_history[offset][-i:]
        true = pred - err
        return pred, true, err

    def log_pred_p_summary(self, offset):
        i = self._last_half_index()
        return self.prediction_history[offset][-i:].mean()

    def entropy_summary(self, offset):
        i = self._last_half_index()
        return np.nanmean(self.entropy_history[offset][-i:])


class AnimationManager:
    # todo: this could inherit from FileWriter; that might be better design
    n_rows = 2
    n_cols = 2
    fps = 20
    dpi = 100
    extension = "mp4"
    outfile = f"./movie.{extension}"
    figsize = (10, 10)

    def __init__(self):
        self.movie_writer = FFMpegFileWriter(fps=self.fps)
        self.fig, self.ax = plt.subplots(self.n_rows, self.n_cols, figsize=self.figsize, layout='tight', squeeze=False)
        self.movie_writer.setup(self.fig, self.outfile, dpi=self.dpi)
        self.finished = False
        self.final_output_location = None
        self.setup()

    def setup(self):
        pass

    def set_final_output_location(self, final_output_location):
        self.final_output_location = final_output_location

    def finish(self):
        if not self.finished:
            self.movie_writer.finish()
            os.rename(self.outfile, self.final_output_location)
            self.finished = True

    def frame_draw_condition(self, step_number, bw):
        return True

    def draw_frame(self, step, bw, br):
        self.custom_draw_frame(step, bw, br)
        self.movie_writer.grab_frame()

    def custom_draw_frame(self, step, bw, br):
        pass
