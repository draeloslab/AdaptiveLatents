# Adaptive Latents project codebase

## Quickstart
```bash
# download the repo
git clone https://github.com/draeloslab/AdaptiveLatents
cd AdaptiveLatents

# install dependencies
conda env create --file=environment.yml
# run the following line if you want to use the GPU version of jax
# python -m "pip" install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda activate adaptive_latents

# install the repo locally (this is optional)
mkdir dist
pip install -e .

# test if everything is working
pytest .  # (optional)
python scripts/main.py
```


## Requirements
Our algorithm is implemented in python with some extra packages including: numpy, jax (for GPU), and matplotlib (for plotting). 

We used python version 3.9 and conda to install the libraries listed in the environment file. 
We provide an environment file for use with conda to create a new environment with these requirements, though we note that jax requires additional setup for GPU integration (https://github.com/google/jax). 



## Refrence
The core Bubblewrap algorithm was initially described here: ['Bubblewrap: Online tiling and real-time flow prediction on neural manifolds'](https://proceedings.neurips.cc/paper/2021/hash/307eb8ee16198da891c521eca21464c1-Abstract.html).
