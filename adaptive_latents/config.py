import yaml
import pathlib
from importlib.resources import files
import functools
import inspect
from frozendict import frozendict
import copy

CONFIG_FILE_NAME = "adaptive_latents_config.yaml"


def merge_dicts(d1, d2):
    # d2 has higher precedence
    common_keys = set(d1.keys()).intersection(d2.keys())
    d3 = d1 | d2
    for key in filter(lambda x: isinstance(x, dict), common_keys):
        d3[key] = merge_dicts(d1[key], d2[key])
    return d3


def freeze_recursively(o):
    match o:
        case dict():
            return frozendict({k: freeze_recursively(v) for k, v in o.items()})
        case list():
            return tuple([freeze_recursively(x) for x in o])
        case _:
            assert type(o) in [str, int, float, bool] or isinstance(o, pathlib.Path)
            return o


def load_config(file, path_keys):
    with open(file, 'r') as fhan:
        config = yaml.safe_load(fhan)

    for key in set(path_keys).intersection(config.keys()):
        config[key] = (file.parent / pathlib.Path(config[key])).resolve()

    return config


def get_config():
    path_keys = ['bwrun_save_path', 'cache_path']
    config = load_config(pathlib.Path(files('adaptive_latents')) / CONFIG_FILE_NAME, path_keys)

    local_config = {}
    for path in pathlib.Path.cwd().parents:
        file = path / CONFIG_FILE_NAME
        if file.is_file():
            local_config = load_config(file, path_keys)
            break

    config = merge_dicts(config, local_config)


    return freeze_recursively(config)


CONFIG = get_config()


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
        config_defaults: dict = copy.deepcopy(CONFIG['default_parameters'][name])

        assert set(current_defaults.keys()) == set(config_defaults.keys())

        config_defaults = config_defaults | kwargs

        return func(*args, **config_defaults)

    return wrapper
