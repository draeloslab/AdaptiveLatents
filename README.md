# Adaptive Latents project codebase

## Quickstart
```bash
# download the repo
git clone https://github.com/draeloslab/AdaptiveLatents
cd AdaptiveLatents

# install dependencies
conda env create --file=environment.yml

# install the repo locally
conda activate adaptive_latents
pip install -e .

# run the following line if you want to use the GPU version of jax
pip install -U "jax[cuda12]"
# note: if using apple silicon, you may need to build jax from source

# create an adaptive_latents_config.yaml file if you wish
# the default copy is found in adaptive_latents/adaptive_latents_config.yaml
# just make sure your copy is in the current working directory when you import adaptive latents

# test if everything is working
python workspace/main.py # this should produce a gif in the current working directory
pytest .
```


## Requirements
Our algorithm is implemented in python with some extra packages including: numpy, jax (for GPU), and matplotlib (for plotting). 

We used python version 3.9 and conda to install the libraries listed in the environment file. 
We provide an environment file for use with conda to create a new environment with these requirements, though we note that jax requires additional setup for GPU integration (https://github.com/google/jax). 

# Development
```bash
# check test code coverage
coverage run -m pytest --longrun
coverage html
open tests/reports/coverage-html/index.html 2>/dev/null

# run the yapf formatter
yapf --in-place --recursive .
```


## Refrence
The core Bubblewrap algorithm was initially described here: ['Bubblewrap: Online tiling and real-time flow prediction on neural manifolds'](https://proceedings.neurips.cc/paper/2021/hash/307eb8ee16198da891c521eca21464c1-Abstract.html).

[//]: # (## Refactor plan)

[//]: # (* the bubblewrap class should have a static run method in addition to BWRun)

[//]: # (* defalut_parameters and the default adaptive_latents_config.yaml should be united &#40;with a default?&#41;)

[//]: # (* I should move the filesystem-dependent files to a new repo or at least the scripts folder)

[//]: # (* move the `dataset_plots` notebook to a function)

[//]: # (* general pipeline component interface so we can share a logger?)

[//]: # (  * learn from sklearn?)

[//]: # (* log caching &#40;and make it local&#41;)

[//]: # (* make Indy dataset not have trailing zeros)

[//]: # (* src/ structure)

[//]: # (* hashable TimedDataSource?)