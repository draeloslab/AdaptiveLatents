from adaptive_latents import Bubblewrap
import datetime
import pickle
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegFileWriter
import warnings
import time
from .config import CONFIG
from types import SimpleNamespace

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .regressions import OnlineRegressor
    from .input_sources.timed_data_source import NumpyTimedDataSource

class BWRun:
    def __init__(self, bw, in_ds,  out_ds=None, behavior_regressor=None, animation_manager=None, log_level=1, show_tqdm=True,
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

        self.show_tqdm = show_tqdm

        self.bw_timepoint_history = []
        self.reg_timepoint_history = []

        self.runtime = None
        self.runtime_since_init = None
        self.hit_end_of_dataset = False


        self.log_level = log_level
        self.add_lambda_functions()
        self.model_offset_variable_history = {key: {offset: [] for offset in in_ds.time_offsets} for key in self.model_offset_variables_to_track}
        self.output_offset_variable_history = {key: {offset: [] for offset in in_ds.time_offsets} for key in self.output_offset_variables_to_track}
        self.model_step_variable_history = {key:[] for key in self.model_step_variables_to_track}
        self.output_step_variable_history = {key:[] for key in self.output_step_variables_to_track}

        self.saved = False
        self.frozen = False

        obs_dim = self.input_ds.output_shape
        assert obs_dim == self.bw.d
        if self.output_regressor:
            assert self.output_ds.output_shape == self.output_regressor.output_d
        # note that if there is no behavior, the behavior dimensions will be zero

    def add_lambda_functions(self):
        self.model_offset_variables_to_track = {}
        self.model_step_variables_to_track = {}
        self.output_step_variables_to_track = {}
        self.output_offset_variables_to_track = {}

        if self.log_level >=0:
            self.model_offset_variables_to_track.update({
                "log_pred_p": lambda bw, o, offset, _: bw.pred_ahead(bw.logB_jax(o, bw.mu, bw.L, bw.L_diag), bw.A, bw.alpha, offset),
                "entropy": lambda bw, o, offset, _: bw.get_entropy(bw.A, bw.alpha, offset),
                "alpha_prediction": lambda bw, o, offset, _: bw.alpha @ np.linalg.matrix_power(bw.A, offset),
            })
            self.model_step_variables_to_track.update({
                "alpha": lambda bw, _: bw.alpha,
            })

        if self.log_level >= 1:
            self.model_step_variables_to_track.update({
                "A": lambda bw, _: bw.A,
                "mu": lambda bw, _: bw.mu,
                "L": lambda bw, _: bw.L,
                "Q": lambda bw, _: bw.Q,
                "n_obs": lambda bw, d: bw.n_obs,
                "n_dead": lambda bw, d: len(bw.dead_nodes),
            })

            self.output_offset_variables_to_track.update({
                "beh_pred": None,
                "beh_error": None
            })
            # todo: make these lambdas
            # todo: make a test to check we can get the error of the beh prediction


        if self.log_level >= 2:
            self.model_step_variables_to_track.update({
                "B": lambda bw, _: bw.B,
                "L_lower": lambda bw, _: bw.L_lower,
                "L_lower_m": lambda bw, _: bw.m_L_lower,
                "L_lower_v": lambda bw, _: bw.v_L_lower,
                "L_lower_grad": lambda bw, _: bw.grad_L_lower,

                "L_diag": lambda bw, _: bw.L_diag,
                "L_diag_m": lambda bw, _: bw.m_L_diag,
                "L_diag_v": lambda bw, _: bw.v_L_diag,
                "L_diag_grad": lambda bw, _: bw.grad_L_diag,

                "pre_B": lambda bw, d: bw.logB_jax(d['offset_pairs'][1], bw.mu, bw.L, bw.L_diag),
            })



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
                        if self.output_regressor:
                            self.reg_timepoint_history.append(self.output_ds.current_timepoint())
                            self.output_regressor.safe_observe(self.bw.alpha, beh)
                            for offset in self.output_ds.time_offsets:
                                b = self.output_ds.get_atemporal_data_point(offset)
                                alpha_ahead = self.bw.alpha @ np.linalg.matrix_power(self.bw.A, offset)
                                bp = self.output_regressor.predict(alpha_ahead)

                                self.output_offset_variable_history["beh_pred"][offset].append(np.array(bp))
                                self.output_offset_variable_history["beh_error"][offset].append(np.array(bp - b))


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
                self.model_offset_variable_history[key][offset].append(np.array(f(self.bw, o, offset, d)))

        for key, f in self.model_step_variables_to_track.items():
            d = dict(offset_pairs=offset_pairs)
            self.model_step_variable_history[key].append(np.array(f(self.bw, d)))

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


        self.bw_timepoint_history = np.array(self.bw_timepoint_history)
        self.reg_timepoint_history = np.array(self.reg_timepoint_history)

        self.model_offset_variable_history = {k: convert_dict(v) for k, v in self.model_offset_variable_history.items()}
        self.output_offset_variable_history = {k: convert_dict(v) for k, v in self.output_offset_variable_history.items()}

        self.model_step_variable_history = convert_dict(self.model_step_variable_history)
        self.output_step_variable_history = convert_dict(self.output_step_variable_history)

        self.h = SimpleNamespace(
            **self.model_step_variable_history,
            **self.model_offset_variable_history,
            **self.output_step_variable_history,
            **self.output_offset_variable_history,
        )


        self.bw.freeze()

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
