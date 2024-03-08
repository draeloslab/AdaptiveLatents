import yaml
import os
import pathlib

config_file_name = "adaptive_latents_config.yaml"

def get_raw_config():
    pwd = pathlib.Path(os.curdir).resolve()
    for path in [pwd] + list(pwd.parents):
        file_candidate = path / config_file_name
        if os.path.exists(file_candidate) and os.path.isfile(file_candidate):
            with open(file_candidate, 'r') as fhan:
                return yaml.safe_load(fhan), path
    raise Exception("No config file found.")


def make_paths_absolute(config, path):
    # todo: this is hacky, but it works for now
    new_config = {}
    for key, value in config.items():
        if type(value) == str:
            p = pathlib.Path(value)
            if not p.is_absolute():
                p = path/p

            new_config[key] = p
        else:
            new_config[key] = value
    return new_config

CONFIG=make_paths_absolute(*get_raw_config())