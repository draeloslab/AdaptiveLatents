import yaml
import pathlib
from importlib.resources import files

CONFIG_FILE_NAME = "adaptive_latents_config.yaml"

def load_config(path):
    file = path / CONFIG_FILE_NAME
    config = {}
    if file.is_file():
        with open(file, 'r') as fhan:
            config = yaml.safe_load(fhan)


    return config

def merge_dicts(d1, d2):
    # d2 has higher precedence
    common_keys = set(d1.keys()).intersection(d2.keys())
    d3 = d1 | d2
    for key in  filter(lambda x: isinstance(x, dict), common_keys):
        d3[key] = merge_dicts(d1[key], d2[key])
    return d3

def get_config():
    config = load_config(pathlib.Path(files('adaptive_latents')))
    local_config = load_config(pathlib.Path.cwd())

    config = merge_dicts(config, local_config)


    for key in ['bwrun_save_path', 'cache_path']:
        config[key] = pathlib.Path(config[key]).resolve()


    return config

CONFIG=get_config()