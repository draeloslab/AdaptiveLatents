import numpy as np

from adaptive_latents import datasets, proSVD, Pipeline, CenteringTransformer, StreamingKalmanFilter, Bubblewrap

from adaptive_latents.predictor import Predictor


class MultiPredictor(Predictor):
    def __init__(self, sub_predictors=(), input_streams=None, output_streams=None, log_level=None, check_dt=False, n_steps_to_predict=1):
        super().__init__(input_streams, output_streams, log_level, check_dt, n_steps_to_predict)
        self.sub_predictors = sub_predictors

    def predict(self, n_steps):
        predictions = []
        for predictor in self.sub_predictors:
            prediction = predictor.predict(n_steps)
            predictions.append(prediction)

            if np.isnan(prediction).any():
                return np.array([[np.nan]])

        return np.array(predictions)

    def unevaluated_log_pred_p(self, n_steps):
        log_pred_ps = []
        for predictor in self.sub_predictors:
            log_pred_p = predictor.unevaluated_log_pred_p(n_steps)
            log_pred_ps.append(log_pred_p)
        def f(x):
            ret = []
            for f in log_pred_ps:
                ret.append(f(x))
            return ret
        return f

    def observe(self, X, stream=None):
        for predictor in self.sub_predictors:
            predictor.observe(X, stream=stream)

    def get_state(self):
        state = []
        for predictor in self.sub_predictors:
            state.append(np.reshape(predictor.get_state(), -1))
        return np.hstack(state) if state else np.array([])

    def get_arbitrary_dynamics_parameter(self):
        dynamics_params = []
        for predictor in self.sub_predictors:
            dynamics_params.append(np.reshape(predictor.get_arbitrary_dynamics_parameter(), -1))
        return np.hstack(dynamics_params)



if __name__ == '__main__':
    d = datasets.Odoherty21Dataset()
    pred = MultiPredictor([StreamingKalmanFilter(), Bubblewrap()], log_level=5, check_dt=True)

    c = CenteringTransformer()
    svd = proSVD(k=10)

    p = Pipeline([c, svd, pred])

    p.offline_run_on(d.neural_data, exit_time=50, show_tqdm=True)