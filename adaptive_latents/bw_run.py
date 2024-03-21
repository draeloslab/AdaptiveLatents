from adaptive_latents import Bubblewrap
import datetime
import pickle
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegFileWriter
from .input_sources.timed_data_source import NumpyTimedDataSource
import warnings
import time
from .config import CONFIG
from types import SimpleNamespace

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .regressions import OnlineRegressor

class BWRun:
    def __init__(self, bw, in_ds, out_ds=None, behavior_regressor=None, animation_manager=None, save_A=False, show_tqdm=True,
                 output_directory=CONFIG["output_path"]/"bubblewrap_runs", notes=()):

        self.bw: Bubblewrap = bw
        self.animation_manager: AnimationManager = animation_manager
        self.input_ds:NumpyTimedDataSource = in_ds
        self.output_ds = out_ds
        # todo: check for beh and obs remnants

        if self.input_ds.output_shape > 10:
            warnings.warn("Bubblewrap might not run well on high-D inputs. Consider using proSVD.")

        self.notes = list(notes)

        # only keep a behavior regressor if there is behavior
        self.output_regressor = None
        if self.output_ds and self.output_ds.output_shape > 0:
            self.output_regressor: OnlineRegressor = behavior_regressor

        self.output_directory = output_directory
        self.create_new_filenames()

        # self.total_runtime = None
        self.save_A = save_A
        self.show_tqdm = show_tqdm

        self.prediction_history = {k: [] for k in in_ds.time_offsets}
        self.entropy_history = {k: [] for k in in_ds.time_offsets}
        self.behavior_pred_history = {k: [] for k in in_ds.time_offsets}
        self.behavior_error_history = {k: [] for k in in_ds.time_offsets}
        self.alpha_history = {k: [] for k in in_ds.time_offsets}
        self.bw_timepoint_history = []
        self.reg_timepoint_history = []
        self.runtime = None
        self.runtime_since_init = None
        self.hit_end_of_dataset = False

        self.n_living_history = []

        self.add_lambda_functions()
        self.model_offset_variable_history = {key: {offset: [] for offset in in_ds.time_offsets} for key in self.model_offset_variables_to_track}
        self.model_step_variable_history = {key:[] for key in self.model_step_variables_to_track}


        if save_A:
            self.A_history = []
            self.B_history = []
            self.pre_B_history = []

            self.mu_history = []
            self.L_history = []
            self.L_lower_history = []
            self.L_diag_history = []
            self.L_diag_m_history = []
            self.L_diag_v_history = []
            self.L_diag_grad_history = []

            self.L_lower_m_history = []
            self.L_lower_v_history = []
            self.L_lower_grad_history = []


        self.saved = False
        self.frozen = False

        obs_dim = self.input_ds.output_shape
        assert obs_dim == self.bw.d
        if self.output_regressor:
            assert self.output_ds.output_shape == self.output_regressor.output_d
        # note that if there is no behavior, the behavior dimensions will be zero

    def add_lambda_functions(self):
        self.model_offset_variables_to_track = {
            "log_pred_p": lambda bw, o, offset, _: bw.pred_ahead(bw.logB_jax(o, bw.mu, bw.L, bw.L_diag), bw.A, bw.alpha, offset),
            "entropy": lambda bw, o, offset, _: bw.get_entropy(bw.A, bw.alpha, offset),
            "alpha_prediction": lambda bw, o, offset, _: bw.alpha @ np.linalg.matrix_power(bw.A, offset),
            # "output_prediction": lambda bw, o, offset, _: ...,
        }
        # todo: make a good way to get behavior error without tracking it


        self.model_step_variables_to_track = {
            "alpha": lambda bw, _: bw.alpha,

            "A": lambda bw, _: bw.A,
            "B": lambda bw, _: bw.B,
            "mu": lambda bw, _: bw.mu,
            "L": lambda bw, _: bw.L,
            "Q": lambda bw, _: bw.Q,

            "L_lower": lambda bw, _: bw.L_lower,
            "L_lower_m": lambda bw, _: bw.m_L_lower,
            "L_lower_v": lambda bw, _: bw.v_L_lower,
            "L_lower_grad": lambda bw, _: bw.grad_L_lower,

            "L_diag": lambda bw, _: bw.L_diag,
            "L_diag_m": lambda bw, _: bw.m_L_diag,
            "L_diag_v": lambda bw, _: bw.v_L_diag,
            "L_diag_grad": lambda bw, _: bw.grad_L_diag,

            "pre_B": lambda bw, d: bw.logB_jax(d['offset_pairs'][1], bw.mu, bw.L, bw.L_diag),
            # "n_living":...,
        }

        self.output_step_variables_to_track = dict()
        self.output_offset_variables_to_track = dict()

    def run(self, save=False, limit=None, freeze=True, initialize=True):
        start_time = time.time()
        time_since_init = None

        if len(self.input_ds) < self.bw.M:
            warnings.warn("Data length shorter than initialization.")

        if limit is None:
            limit = len(self.input_ds)
        limit = min(len(self.input_ds), limit)

        bw_step = 0
        obs_next_t, obs_done = self.input_ds.preview_next_timepoint()
        if self.output_ds:
            beh_next_t, beh_done = self.output_ds.preview_next_timepoint()
        else:
            beh_next_t, beh_done = float("inf"), True

        with tqdm(total=limit, disable=not self.show_tqdm) as pbar:
            while not (obs_done and beh_done) and bw_step <= limit:
                if beh_done or obs_next_t < beh_next_t:
                    obs = next(self.input_ds)
                    self.bw.observe(obs)
                    obs_next_t, obs_done = self.input_ds.preview_next_timepoint()

                    if initialize and bw_step < self.bw.M:
                        bw_step += 1
                        pbar.update(1)
                        continue
                    elif initialize and bw_step == self.bw.M:
                        self.bw.init_nodes()
                        self.bw.e_step()  # todo: is this OK?
                        self.bw.grad_Q()
                        time_since_init = time.time()
                    else:
                        self.bw.e_step()
                        self.bw.grad_Q()

                    pairs = {}
                    for offset in self.input_ds.time_offsets:
                        pairs[offset] = self.input_ds.get_atemporal_data_point(offset)
                    self.bw_log_for_step(bw_step, pairs)
                    bw_step += 1
                    pbar.update(1)
                else:
                    beh = next(self.output_ds)
                    beh_next_t, beh_done = self.output_ds.preview_next_timepoint()
                    if hasattr(self.bw, 'alpha'):
                        self.reg_timepoint_history.append(self.output_ds.current_timepoint())
                        if self.output_regressor:
                            self.output_regressor.safe_observe(self.bw.alpha, beh)

                            for offset in self.output_ds.time_offsets:
                                b = self.output_ds.get_atemporal_data_point(offset)
                                alpha_ahead = self.alpha_history[offset][-1]
                                bp = self.output_regressor.predict(alpha_ahead)

                                self.behavior_pred_history[offset].append(bp)
                                self.behavior_error_history[offset].append(bp - b)



        end_time = time.time()
        self.runtime_since_init = (end_time - time_since_init) if time_since_init is not None else np.nan
        self.runtime = end_time - start_time
        self.hit_end_of_dataset = obs_done and beh_done

        if freeze:
            self.finish_and_remove_jax()
        if save:
            self.saved = True
            with open(self.pickle_file, "wb") as fhan:
                pickle.dump(self, fhan)

    def bw_log_for_step(self, step, offset_pairs):
        # TODO: allow skipping of (e.g. entropy) steps?
        for offset, o in offset_pairs.items():
            for key, f in self.model_offset_variables_to_track.items():
                d = dict()
                self.model_offset_variable_history[key][offset].append(f(self.bw, o, offset, d))

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
            self.pre_B_history.append(self.bw.logB_jax(offset_pairs[1], self.bw.mu, self.bw.L, self.bw.L_diag))
            self.B_history.append(self.bw.B)
            self.mu_history.append(self.bw.mu)
            self.L_history.append(self.bw.L)
            self.L_diag_history.append(self.bw.L_diag)
            self.L_diag_m_history.append(np.array(self.bw.m_L_diag))
            self.L_diag_v_history.append(np.array(self.bw.v_L_diag))
            self.L_diag_grad_history.append(np.array(self.bw.grad_L_diag))

            self.L_lower_history.append(self.bw.L_lower)
            self.L_lower_m_history.append(np.array(self.bw.m_L_lower))
            self.L_lower_v_history.append(np.array(self.bw.v_L_lower))
            self.L_lower_grad_history.append(np.array(self.bw.grad_L_lower))

            for key, f in self.model_step_variables_to_track.items():
                d = dict(offset_pairs=offset_pairs)
                self.model_step_variable_history[key].append(f(self.bw, d))



        if self.animation_manager and self.animation_manager.frame_draw_condition(step, self.bw):
            self.animation_manager.draw_frame(step, self.bw, self)

        self.bw_timepoint_history.append(self.input_ds.current_timepoint())

    def create_new_filenames(self):
        time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        self.output_prefix = os.path.join(self.output_directory, f"bubblewrap_run_{time_string}")
        self.pickle_file = f"{self.output_prefix}.pickle"
        if self.animation_manager:
            self.animation_manager.set_final_output_location(f"{self.output_prefix}.{self.animation_manager.extension}")

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
            self.pre_B_history = np.array(self.pre_B_history)
            self.B_history = np.array(self.B_history)
            self.mu_history = np.array(self.mu_history)
            self.L_history = np.array(self.L_history)

            self.L_diag_history = np.array(self.L_diag_history)
            self.L_diag_m_history = np.array(self.L_diag_m_history)
            self.L_diag_v_history = np.array(self.L_diag_v_history)
            self.L_diag_grad_history = np.array(self.L_diag_grad_history)

            self.L_lower_history = np.array(self.L_lower_history)
            self.L_lower_m_history = np.array(self.L_lower_m_history)
            self.L_lower_v_history = np.array(self.L_lower_v_history)
            self.L_lower_grad_history = np.array(self.L_lower_grad_history)

            self.model_step_variable_history = convert_dict(self.model_step_variable_history)

        self.h = SimpleNamespace(
            **self.model_step_variable_history,
            **self.model_offset_variable_history
        )


        self.bw.freeze()

    # Metrics
    def evaluate_regressor(self, reg, o=None, o_t=None, train_offset=0, test_offset=1):
        warnings.warn("this method isn't tested well, use at your own risk")
        if o is None and o_t is None:
            o = self.output_ds.a
            o_t = self.output_ds.t

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

    def _last_half_index(self, offset=1):
        warnings.warn("this method is outdated for non-synchronous data")
        assert offset in self.input_ds.time_offsets
        assert self.frozen
        assert self.input_ds.time_offsets
        pred = self.prediction_history[offset]
        assert np.all(np.isfinite(pred)) # this works because there are no skipped steps
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


    def __getstate__(self):
        d = self.__dict__
        expected_variables = {"model_offset_variables_to_track", "model_step_variables_to_track", "output_step_variables_to_track", "output_offset_variables_to_track"}
        assert {x for x in d if "variables_to_track" in x} == expected_variables

        for variable in expected_variables:
            temp = {key:"removed for pickling" for key, value in d[variable].items()}
            d[variable] = temp
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.add_lambda_functions()


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
