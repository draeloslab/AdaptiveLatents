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

def get_config():
    config = load_config(pathlib.Path(files('adaptive_latents')))
    local_config = load_config(pathlib.Path.cwd())

    # get_config assumes there's no hierarchy when we merge
    assert all(map(lambda x: not isinstance(x, dict), [*config.values(), *local_config.values()] ))

    config.update(local_config)


    config['output_path'] = pathlib.Path(config['output_path']).resolve()
    config['data_path'] = pathlib.Path(config['data_path']).resolve()


    return config

CONFIG=get_config()