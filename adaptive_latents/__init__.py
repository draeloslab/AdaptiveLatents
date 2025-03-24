import jax

from .config import CONFIG

jax.config.update('jax_enable_x64', CONFIG.jax_enable_x64)
jax.config.update('jax_platform_name', CONFIG.jax_platform_name)

from . import input_sources, plotting_functions, profiling_functions, utils, predictor
from .bubblewrap import Bubblewrap
from .ica import mmICA
from .input_sources import datasets
from .jpca import sjPCA
from .plotting_functions import AnimationManager
from .pro_pls import proPLS
from .prosvd import RandomProjection, proSVD
from .regressions import VanillaOnlineRegressor
from .timed_data_source import ArrayWithTime
from .transformer import CenteringTransformer, Concatenator, KernelSmoother, Pipeline, Tee, ZScoringTransformer
from .vjf import VJF
from .input_sources.kalman_filter import StreamingKalmanFilter