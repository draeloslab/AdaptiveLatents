import copy
import functools
import inspect
import pathlib

import importlib_resources as impresources
import yaml


class ConfigObject:
    CONFIG_FILE_NAME = "adaptive_latents_config.yaml"
    n_instances = 0

    def __init__(self, base_config=None, local_config=None):
        self.n_instances += 1
        assert self.n_instances == 1, "This class was originally designed to be a single global module object. Make more at your own risk."

        if base_config is None:
            base_config, local_config = self.default_search()

        simple_combination = base_config | local_config

        self.jax_enable_x64 = simple_combination["jax_enable_x64"]
        self.jax_platform_name = simple_combination["jax_platform_name"]

        self.supress_dandi_warnings = simple_combination["supress_dandi_warnings"]

        self.attempt_to_cache = simple_combination["attempt_to_cache"]
        self.verbose = simple_combination["verbose"]

        self.bwrun_save_path = self.resolve_path(base_config, local_config, "bwrun_save_path")
        self.plot_save_path = self.resolve_path(base_config, local_config, "plot_save_path")
        self.cache_path = self.resolve_path(base_config, local_config, "cache_path")
        self.dataset_path = self.resolve_path(base_config, local_config, "dataset_path")


        self.default_parameters = {}
        for key, params in base_config['default_parameters'].items():
            local_params = local_config.get(key, {})
            self.default_parameters[key] = params | local_params

    @staticmethod
    def resolve_path(base_config, local_config, key):
        if key in local_config:
            path =pathlib.Path(local_config['config_file_directory'] / local_config[key])
        else:
            path = pathlib.Path(base_config[key])

        return path

    def default_search(self):
        with open(pathlib.Path(impresources.files('adaptive_latents')) / self.CONFIG_FILE_NAME, 'r') as fhan:
            base_config = yaml.safe_load(fhan)

        local_config = {}
        cwd = pathlib.Path.cwd()
        for path in [cwd] + list(cwd.parents):
            file = path / self.CONFIG_FILE_NAME
            if file.is_file():
                with open(file, 'r') as fhan:
                    local_config = yaml.safe_load(fhan)
                    local_config['config_file_directory'] = path
                break

        return base_config, local_config

    @staticmethod
    def open_with_parents(file, mode, open_kwargs=None):
        # TODO: it's inelegant to have to remember to use this; replace somehow?
        open_kwargs = open_kwargs or {}
        pathlib.Path(file).parent.mkdir(exist_ok=True, parents=True)
        return open(file, mode, **open_kwargs)


CONFIG = ConfigObject()


def use_config_defaults(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # https://stackoverflow.com/a/12627202
        sig = inspect.signature(func)
        current_defaults = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}

        name = func.__qualname__
        # this is a little hacky but it works
        if "__init__" in name:
            name = name.split(".")
            assert len(name) == 2
            name = name[0]
        config_defaults: dict = copy.deepcopy(CONFIG.default_parameters[name])

        assert set(current_defaults.keys()) == set(config_defaults.keys())

        config_defaults = config_defaults | kwargs

        return func(*args, **config_defaults)

    return wrapper
