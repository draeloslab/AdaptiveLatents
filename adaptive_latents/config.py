import yaml
import pathlib
from importlib.resources import files
import functools
import inspect
from types import MappingProxyType

CONFIG_FILE_NAME = "adaptive_latents_config.yaml"


def merge_dicts(d1, d2):
    # d2 has higher precedence
    common_keys = set(d1.keys()).intersection(d2.keys())
    d3 = d1 | d2
    for key in filter(lambda x: isinstance(x, dict), common_keys):
        d3[key] = merge_dicts(d1[key], d2[key])
    return d3

def load_config(path):
    file = path / CONFIG_FILE_NAME
    config = {}
    if file.is_file():
        with open(file, 'r') as fhan:
            config = yaml.safe_load(fhan)
    return config


def get_config():
    config = load_config(pathlib.Path(files('adaptive_latents')))
    local_config = load_config(pathlib.Path.cwd())

    config = merge_dicts(config, local_config)

    for key in ['bwrun_save_path', 'cache_path']:
        config[key] = pathlib.Path(config[key]).resolve()

    d = config['default_parameters']['Bubblewrap']
    # this is inelegant
    for key in d:
        try:
            d[key] = float(d[key])
            if d[key] % 1 == 0:
                d[key] = int(d[key])
        except ValueError:
            d[key] = bool(d[key])
    d['seed'] = int(d['seed'])


    return config

CONFIG=get_config()

def use_config_defaults(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # https://stackoverflow.com/a/12627202
        sig = inspect.signature(func)
        current_defaults = {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        name = func.__qualname__
        # this is a little hacky but it works
        if "__init__" in name:
            name = name.split(".")
            assert len(name) == 2
            name = name[0]
        config_defaults: dict = CONFIG['default_parameters'][name]

        assert set(current_defaults.keys()) == set(config_defaults.keys())

        config_defaults = config_defaults | kwargs

        return func(*args, **config_defaults)
    return wrapper