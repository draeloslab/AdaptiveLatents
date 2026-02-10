import jax

from .config import CONFIG

if CONFIG.jax_supress_xla_bridge_warnings:
    import logging
    logging.getLogger('jax._src.lib.xla_bridge').addFilter(lambda _: False)

jax.config.update('jax_enable_x64', CONFIG.jax_enable_x64)
jax.config.update('jax_platform_name', CONFIG.jax_platform_name)

from . import input_sources, plotting_functions, predictor, profiling_functions, utils
from .bubblewrap import Bubblewrap
from .input_sources.kalman_filter import StreamingKalmanFilter
from .jpca import sjPCA
from .plotting_functions import AnimationManager
from .pro_pls import proPLS
from .prosvd import RandomProjection, proSVD
from .regressions import VanillaOnlineRegressor, BaseMultiKernelRegressor
from .stim_regressor import StimRegressor
from .timed_data_source import ArrayWithTime
from .transformer import CenteringTransformer, Concatenator, KernelSmoother, Pipeline, Tee, ZScoringTransformer
from . import sim_stim

try:
    from .input_sources import datasets
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .ica import mmICA
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .vjf import VJF
except (ImportError, ModuleNotFoundError):
    pass

# TODO: remove https://kmichel.github.io/python-importtime-graph/
