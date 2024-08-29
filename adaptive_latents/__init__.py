from .config import CONFIG

import jax
jax.config.update('jax_enable_x64', CONFIG['jax_enable_x64'])
jax.config.update('jax_platform_name', CONFIG['jax_platform_name'])

from .bubblewrap import Bubblewrap
from .bw_run import BWRun, AnimationManager
from .regressions import VanillaOnlineRegressor, SemiRegularizedRegressor
from .timed_data_source import NumpyTimedDataSource
from .prosvd import proSVD, RandomProjection
from .ica import mmICA
from .jpca import sjPCA
from .pro_pls import proPLS
from .transformer import CenteringTransformer, Pipeline, KernelSmoother
from . import plotting_functions
from . import profiling_functions
from . import utils
from . import input_sources
