from .config import CONFIG

import jax
jax.config.update('jax_enable_x64', CONFIG['jax_enable_x64'])
jax.config.update('jax_platform_name', CONFIG['jax_platform_name'])

from .bubblewrap import Bubblewrap
from .bw_run import BWRun, AnimationManager
from .default_parameters import default_rwd_parameters
from .regressions import VanillaOnlineRegressor, SemiRegularizedRegressor
from .input_sources import NumpyTimedDataSource
from . import plotting_functions
from .transforms import proSVD, sjPCA
from . import profiling_functions
