import pathlib
import yaml

def log_for_tex(*, key, value, current_file, output_directory):
    stem = pathlib.Path(current_file).stem
    output_file = output_directory / f'generated_by_{stem}.yaml'

    if output_file.exists():
        with open(output_file, 'r') as fhan:
            existing_entries = yaml.safe_load(fhan)
    else:
        existing_entries = {}

    if existing_entries is None:
        existing_entries = {}

    existing_entries.update({key: value})

    with open(output_file, 'w') as fhan:
        yaml.safe_dump(existing_entries, fhan)

    return value