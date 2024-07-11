from adaptive_latents import Bubblewrap, CONFIG
import datetime
import pickle
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegFileWriter
import warnings
import time
from types import SimpleNamespace
import pathlib

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .regressions import OnlineRegressor
    from .input_sources.timed_data_source import NumpyTimedDataSource

class BWRun:
    def __init__(self, bw, in_ds,  out_ds=None, behavior_regressor=None, animation_manager=None, log_level=1, show_tqdm=True,
                 output_directory=CONFIG['bwrun_save_path'], notes=()):
        # todo: output_directory in CONFIG

        self.bw: Bubblewrap = bw
        self.animation_manager: AnimationManager = animation_manager
        self.input_ds: NumpyTimedDataSource = in_ds
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

        self.runtime = None
        self.runtime_since_init = None
        self.hit_end_of_dataset = False
        self.bw_init_time = None


        self.log_level = log_level
        self.add_lambda_functions()
        self.model_offset_variable_history = {key: {offset: [] for offset in in_ds.time_offsets} for key in self.model_offset_variables_to_track}
        self.model_step_variable_history = {key:[] for key in self.model_step_variables_to_track}

        if out_ds is not None:
            self.output_offset_variable_history = {key: {offset: [] for offset in out_ds.time_offsets} for key in self.output_offset_variables_to_track}
            self.output_step_variable_history = {key:[] for key in self.output_step_variables_to_track}
        else:
            self.output_offset_variable_history = {}
            self.output_step_variable_history = {}

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
                "bw_offset_t": lambda bw, o, offset, d: d["t"],
                "bw_offset_origin_t": lambda bw, o, offset, d: d["origin_t"],
            })
            self.model_step_variables_to_track.update({
                "alpha": lambda bw, _: bw.alpha,
                "bw_t": lambda bw, d: d["t"],
            })


            self.output_step_variables_to_track.update({
                "last_alpha": None,
                "beta": None,
            })
            self.output_offset_variables_to_track.update({
                "beh_pred": None,
                "beh_error": None,
                "reg_offset_t": None,
                "reg_offset_origin_t": None,
            })
            # TODO: make these lambdas
            # TODO: make a test to check we can get the error of the beh prediction

        if self.log_level >= 1:
            self.model_step_variables_to_track.update({
                "A": lambda bw, _: bw.A,
                "mu": lambda bw, _: bw.mu,
                "L": lambda bw, _: bw.L,
                "Q": lambda bw, _: bw.Q,
                "n_obs": lambda bw, d: bw.n_obs,
                "n_dead": lambda bw, d: len(bw.dead_nodes),
            })



        if self.log_level >= 2:
            self.model_step_variables_to_track.update({
                "Q_parts": lambda bw, _: bw.Q_parts,


                "L_lower": lambda bw, _: bw.L_lower,
                "L_lower_m": lambda bw, _: bw.m_L_lower,
                "L_lower_v": lambda bw, _: bw.v_L_lower,
                "L_lower_grad": lambda bw, _: bw.grad_L_lower,

                "L_diag": lambda bw, _: bw.L_diag,
                "L_diag_m": lambda bw, _: bw.m_L_diag,
                "L_diag_v": lambda bw, _: bw.v_L_diag,
                "L_diag_grad": lambda bw, _: bw.grad_L_diag,

                "B": lambda bw, _: bw.B,
                # "pre_B": lambda bw, d: bw.logB_jax(d['offset_pairs'][1], bw.mu, bw.L, bw.L_diag),

            })




    def run(self, save=False, limit=None, freeze=True):
        start_time = time.time()

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

                    self.next_bubblewrap_step(obs)
                    self.log_for_bw_step(bw_step)

                    bw_step += 1
                    # assert bw_step == self.bw.obs.n_obs
                    pbar.update(1)

                    obs_next_t, obs_done = self.input_ds.preview_next_timepoint()
                else:
                    beh = next(self.output_ds)
                    beh_next_t, beh_done = self.output_ds.preview_next_timepoint()
                    if self.output_regressor and self.bw.is_initialized:
                        self.output_regressor.observe(self.bw.alpha, beh) # todo: this should be historical?
                        self.log_for_regression_step()




        end_time = time.time()
        self.runtime_since_init = (end_time - self.bw_init_time) if self.bw_init_time is not None else np.nan
        self.runtime = end_time - start_time
        self.hit_end_of_dataset = obs_done and beh_done

        if freeze:
            self.finish_and_remove_jax()
        if save:
            self.save()

    def save(self):
        self.saved = True
        pathlib.Path(self.pickle_file).parent.mkdir(exist_ok=True) # todo: put this in config?
        with open(self.pickle_file, "wb") as fhan:
            pickle.dump(self, fhan)

    def next_bubblewrap_step(self, obs):
        self.bw.observe(obs)

        if self.bw.obs.n_obs < self.bw.M:
            ...
        elif self.bw.obs.n_obs == self.bw.M:
            self.bw.init_nodes()
            self.bw.e_step()
            self.bw.grad_Q()
            self.bw_init_time = time.time()
        else:
            self.bw.e_step()
            self.bw.grad_Q()


    def log_for_regression_step(self, at_timepoint=None):
        alpha = self.bw.alpha
        A = self.bw.A
        if at_timepoint is not None:
            idx = np.nonzero(at_timepoint > self.model_step_variable_history['bw_t'])[0]
            if not len(idx):
                raise Exception("An impossible time was requested for a log, possibly in a post-hoc regression.")

            idx = idx[-1]
            alpha = self.model_step_variable_history['alpha'][idx]
            A = self.model_step_variable_history['A'][idx]

        if "last_alpha" in self.output_step_variables_to_track:
            self.output_step_variable_history["last_alpha"].append(alpha)

        if "beta" in self.output_step_variables_to_track and hasattr(self.output_regressor, "get_beta"):
            beta = self.output_regressor.get_beta()
            self.output_step_variable_history["beta"].append(beta)

        for offset in self.output_ds.time_offsets:
            b = self.output_ds.get_atemporal_data_point(offset)
            alpha_ahead = alpha @ np.linalg.matrix_power(A, offset)
            bp = self.output_regressor.predict(alpha_ahead)

            if "beh_pred" in self.output_offset_variables_to_track:
                self.output_offset_variable_history["beh_pred"][offset].append(np.atleast_1d(bp))

            if "beh_error" in self.output_offset_variables_to_track:
                self.output_offset_variable_history["beh_error"][offset].append(np.atleast_1d(bp - b))

            if "reg_offset_t" in self.output_offset_variables_to_track:
                t, _ = self.output_ds.preview_next_timepoint(offset=offset)
                self.output_offset_variable_history["reg_offset_t"][offset].append(t)
                self.output_offset_variable_history["reg_offset_origin_t"][offset].append(self.output_ds.current_timepoint())

    def log_for_bw_step(self, step):
        if self.bw.is_initialized:
            offset_pairs = {}
            for offset in self.input_ds.time_offsets:
                offset_pairs[offset] = self.input_ds.get_atemporal_data_point(offset)

            for offset, o in offset_pairs.items():
                for key, f in self.model_offset_variables_to_track.items():
                    origin_t = self.input_ds.current_timepoint()
                    t, _ = self.input_ds.preview_next_timepoint(offset=offset)
                    d = dict(t=t, origin_t=origin_t)
                    self.model_offset_variable_history[key][offset].append(np.array(f(self.bw, o, offset, d)))

            for key, f in self.model_step_variables_to_track.items():
                d = dict(offset_pairs=offset_pairs, t=self.input_ds.current_timepoint())
                self.model_step_variable_history[key].append(np.array(f(self.bw, d)))

            if self.animation_manager and self.animation_manager.frame_draw_condition(step, self.bw):
                self.animation_manager.draw_frame(step, self.bw, self)

    def add_regression_post_hoc(self, regressor, output_ds):
        assert "alpha" in self.model_step_variable_history
        assert self.output_ds is None
        # assert not self.frozen

        self.output_ds = output_ds
        self.output_regressor = regressor

        self.output_offset_variable_history = {key: {offset: [] for offset in self.output_ds.time_offsets} for key in
                                               self.output_offset_variables_to_track}
        self.output_step_variable_history = {key: [] for key in self.output_step_variables_to_track}

        beh_next_t, beh_done = self.output_ds.preview_next_timepoint()

        while not beh_done and beh_next_t <= self.input_ds.current_timepoint():
            beh = next(self.output_ds)

            idx = np.nonzero(self.output_ds.current_timepoint() > self.model_step_variable_history['bw_t'] )[0]
            if len(idx):
                idx = idx[-1]
                alpha = self.model_step_variable_history['alpha'][idx]
                self.output_regressor.observe(alpha, beh)
                self.log_for_regression_step(at_timepoint=self.output_ds.current_timepoint())

            beh_next_t, beh_done = self.output_ds.preview_next_timepoint()

        if self.frozen:
            self.make_h()

    def get_last_half_time(self, offset):
        bw_t = self.model_step_variable_history['bw_t']
        last_half_time = bw_t[len(bw_t)//2]
        if self.output_regressor:
            err = self.output_offset_variable_history['beh_error'][offset]
            err = np.array(err)
            idx = np.nonzero(~np.all(np.isnan(err), axis=1))[0][0]
            idx = (err.shape[0] - idx) // 2
            lht = self.output_offset_variable_history['reg_offset_origin_t'][offset][err.shape[0] - idx]
            last_half_time = max(last_half_time, lht)

        return last_half_time

    def get_last_half_metrics(self, offset):
        metrics = {}

        halfway_time = self.get_last_half_time(offset)
        if self.output_regressor:
            s = self.output_offset_variable_history['reg_offset_origin_t'][offset] > halfway_time
            metrics['beh_sq_error'] = list(np.mean(np.array(self.output_offset_variable_history['beh_error'][offset])[s]**2,0))
            beh_error = np.array(self.output_offset_variable_history['beh_error'][offset])[s]
            beh_pred = np.array(self.output_offset_variable_history['beh_pred'][offset])[s]
            beh_true = beh_pred - beh_error
            if len(beh_true.shape) == 1:
                # TODO: make this unecessary
                beh_true = beh_true.reshape([-1, 1])
                beh_pred = beh_pred.reshape([-1, 1])
            metrics['beh_corr'] = [np.corrcoef(beh_true[:,j], beh_pred[:,j])[0,1] for j in range(beh_true.shape[1])]

        s = self.model_offset_variable_history['bw_offset_origin_t'][offset] > halfway_time
        metrics['log_pred_p'] = np.mean(self.model_offset_variable_history['log_pred_p'][offset])
        metrics['entropy'] = np.mean(self.model_offset_variable_history['entropy'][offset])

        return metrics


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

        self.make_h()

        self.bw.freeze()

    def make_h(self):
        def convert_dict(d):
            return {k: np.array(v) for k, v in d.items()}


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
