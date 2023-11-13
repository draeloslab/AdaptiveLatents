# Adaptive Latents project codebase

## Quickstart
```bash
git clone https://github.com/draeloslab/bubblewrap
conda env create --file=environment.yml
conda activate bubblewrap
mkdir dist
pip install -e .
pytest .  # (optional)
python scripts/main.py
```


## Requirements
Our algorithm is implemented in python with some extra packages including: numpy, jax (for GPU), and matplotlib (for plotting). 

We used python version 3.9 and conda to install the libraries listed in the environment file. 
We provide an environment file for use with conda to create a new environment with these requirements, though we note that jax requires additional setup for GPU integration (https://github.com/google/jax). 





## Refrence
The core Bubblewrap algorithm was initially described here: ['Bubblewrap: Online tiling and real-time flow prediction on neural manifolds'](https://proceedings.neurips.cc/paper/2021/hash/307eb8ee16198da891c521eca21464c1-Abstract.html).
