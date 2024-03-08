from adaptive_latents import Bubblewrap, BWRun, AnimationManager, SymmetricNoisyRegressor
from adaptive_latents.regressions import NearestNeighborRegressor
from adaptive_latents.default_parameters import default_jpca_dataset_parameters
from adaptive_latents.input_sources.timed_data_source import NumpyDataSource, PairWrapperSource
import numpy as np


def inner_evaluate(parameters, scrap=0):
    obs, beh = NumpyDataSource.get_from_saved_npz("jpca_reduced_sc.npz", time_offsets=(2,))
    ds = PairWrapperSource(obs, beh)
    ds.shorten(scrap)

    bw = Bubblewrap(dim=ds.output_shape[0], **default_jpca_dataset_parameters)

    reg = SymmetricNoisyRegressor(input_d=bw.N, output_d=1, **parameters)

    br = BWRun(bw=bw, data_source=ds, behavior_regressor=reg, show_tqdm=True)

    br.run(limit=1400, save=False)

    steps = 2
    last = br.prediction_history[steps][-300:]
    last_e = br.entropy_history[steps][-300:]

    bp_last = np.array(br.behavior_error_history[steps][-300:]) ** 2
    mse = np.mean(bp_last)

    # if np.any(np.isnan(bp_last)):
    #     mse = np.quantile(a=bp_last[np.isfinite(bp_last)], q=.9)
    # mse = min(mse, 1e5) + (mse % 10)

    return {
        # "bw_log_pred_pr": np.mean(last),
        # "entropy":np.mean(last_e),
        "runtime": br.runtime,
        "regression_mse": mse,
    }


def evaluate(parameters):
    results = {}
    for to_scrap in [0, 25, 50, 100, 150, 175]:
        inner_parameters = dict(**parameters)
        for key, value in inner_evaluate(inner_parameters, scrap=to_scrap).items():
            results[key] = results.get(key, []) + [value]
    for key, value in results.items():
        results[key] = (np.mean(value), np.std(value) / np.sqrt(len(value)))
    return results

inner_evaluate({})


from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
import botorch
from ax.storage.registry_bundle import RegistryBundle


def make_generation_strategy():
    gs = GenerationStrategy(
        steps=[
            # Quasi-random initialization step
            GenerationStep(
                model=Models.SOBOL,
                num_trials=10,  # How many trials should be produced from this generation step
            ),
            # Bayesian optimization step using the custom acquisition function
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,  # No limitation on how many trials should be produced from this step
                # For `BOTORCH_MODULAR`, we pass in kwargs to specify what surrogate or acquisition function to use.
                model_kwargs={
                    # "surrogate": Surrogate(botorch.models.SingleTaskGP),
                    # "botorch_acqf_class": botorch.acquisition.ExpectedImprovement
                },
            ),
        ]
    )
    return gs




def make_ax_client(generation_strategy):
    ax_client = AxClient(generation_strategy=generation_strategy)

    ax_client.create_experiment(
        name="sn_2step_jpca_defaults",
        parameters=[
            {
                "name": "forgetting_factor",
                "type": "range",
                "bounds": [1e-5, .15],
                "value_type": 'float',
                "log_scale": True,
            },
            {
                "name": "noise_scale",
                "type": "range",
                "bounds": [1e-5, 2],
                "value_type": 'float',
                "log_scale": True,
            },
            {
                "name": "n_perturbations",
                "type": "range",
                "bounds": [1, 6],
                "value_type": 'int',
            },
        ],
        objectives={
            "regression_mse": ObjectiveProperties(minimize=True),
        },
        tracking_metric_names=[
            "runtime",
            "entropy",
            "bw_log_pred_pr"
        ]
    )


    return ax_client


def manually_add_old_trials(ax_client, fname):
    old_ax_client = AxClient.load_from_json_file(fname)
    for key, value in old_ax_client.experiment.trials.items():
        df = list(old_ax_client.experiment.data_by_trial[key].values())[0].df
        dd = {}
        for i in df.index:
            dd[df.loc[i, "metric_name"]] = (df.loc[i, "mean"], df.loc[i, "sem"])
        p = value.arm.parameters
        try:
            _, idx = ax_client.attach_trial(p)
        except ValueError:
            continue
        ax_client.complete_trial(idx, dd)


def main():
    gs = make_generation_strategy()
    ax_client = make_ax_client(gs)

    # manually_add_old_trials(ax_client, f"/home/jgould/Documents/Bubblewrap/generated/optim/ax_{ax_client.experiment.name}_snapshot.json")

    for i in range(100):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
        if i % 5 == 1:
            ax_client.save_to_json_file(
                f"/home/jgould/Documents/Bubblewrap/generated/optim/ax_{ax_client.experiment.name}_snapshot.json")


if __name__ == '__main__':
    # IMPORTANT: adaptive_latents has to run once before Ax gets imported or things break
    main()
